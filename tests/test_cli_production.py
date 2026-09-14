"""Production inputs, fixed-update training, checkpoint replay and raw products."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import pytest
import torch

from torchsx.cli import _pipeline as P, train, merge_nn
from tests.test_amortized_training import _synthetic
from tests.test_cli_pipeline import _stack, _DataArgs
from tests.test_measurement_sources import stream_file, python_parser


@pytest.mark.parametrize('module,count', [(train,16), (merge_nn,13)])
def test_public_options_and_split_inputs(module, count):
    parser = module._build_parser()
    opts = [a for a in parser._actions if a.option_strings and '--help' not in a.option_strings]
    assert len(opts) == count
    prefix = [] if module is train else ['model.pt']
    a = parser.parse_args(prefix + ['--split','dark=dark.stream','--split','light=light.stream','-o','run'])
    assert P._splits_from_args(a) == [('dark',Path('dark.stream')),('light',Path('light.stream'))]
    a = parser.parse_args(prefix + ['combined.stream','--split-file','custom.split','-o','run'])
    assert a.split_file == Path('custom.split')


@pytest.mark.parametrize('module', [train,merge_nn])
@pytest.mark.parametrize('option,value', [('--gain','garbage'),('--scatter-family','mixture'),('--calib-degree','9,9')])
def test_invalid_or_removed_options_fail_before_loading(module,option,value,capsys):
    prefix = [] if module is train else ['model.pt']
    with pytest.raises(SystemExit):
        module.main(prefix + ['missing.stream','-o','run',option,value])
    error = capsys.readouterr().err
    assert value in error
    if option == '--gain':
        for valid in ('auto','trust_input','calibrate','positive finite'):
            assert valid in error


def test_custom_split_matches_source_images(stream_file,python_parser,tmp_path):
    labels = tmp_path/'custom.split'
    labels.write_text('/raw/a.h5 //5 dark\n/raw/b.h5 //8 light\n')
    args = _DataArgs(stream=[stream_file],split_file=labels,int_radius=None,gain='trust_input')
    bundle=P.load_prep(args,verbose=False)
    assert bundle.split_names == ['dark','light']
    assert set(bundle.prep.obs_info.delay_bin_idx.tolist()) == {0,1}


def test_cache_moves_with_inputs_and_rejects_changed_content(stream_file,python_parser,tmp_path,monkeypatch):
    cache=tmp_path/'prepared.pt'
    args=_DataArgs(stream=[stream_file],int_radius=None,gain=None,cache_prep=cache)
    built=P.load_prep(args,verbose=False)
    moved=tmp_path/'moved';moved.mkdir()
    source=moved/stream_file.name;shutil.copyfile(stream_file,source)
    relocated=moved/'prepared.pt';shutil.copyfile(cache,relocated)
    args.stream,args.cache_prep=[source],relocated
    monkeypatch.setattr('torchsx.stream.SFXStream.read',lambda *a,**k: pytest.fail('cache hit reparsed observations'))
    loaded=P.load_prep(args,recipe_defaults=built.recipe,verbose=False)
    assert torch.equal(loaded.prep.I_obs,built.prep.I_obs)
    source.write_text(source.read_text()+'\n# changed input\n')
    with pytest.raises(SystemExit,match='different settings'):
        P.load_prep(args,verbose=False)


def test_defaults_and_old_checkpoint_likelihood(tmp_path):
    prep,params,*_=_synthetic(n_hkl=20,n_xtal=40,per=10)
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1'},[],0.125)
    args=train._build_parser().parse_args(['x.stream','-o',str(tmp_path),'--device','cpu'])
    host,_,scale,obj,S=P.build_stack(bundle,args,verbose=False)
    assert host is prep
    assert (scale.heads.scatter_family,scale.heads.n_components,scale.heads.mixture.shape)==('mixture',4,'gaussian')
    assert obj.likelihood.bg_mean_offset==0.125
    oldscale,oldobj,tr=_stack(prep,params,tmp_path)
    try:
        ck=torch.load(tr.save_best(),weights_only=False)
        _,_,rebuilt,oldlik,_=P.build_stack(bundle,args,ckpt=ck,verbose=False)
        rebuilt.load_state_dict(ck['state_dict'])
        assert rebuilt.heads.scatter_family=='gengauss'
        assert oldlik.likelihood.bg_mean_offset==0.0
    finally:
        tr.close()


def test_batches_are_balanced_and_allow_small_states():
    from torchsx.nn.train_amortized import CrystalIndex
    condition=torch.tensor([0,0,1,1,1,1])
    index=CrystalIndex(torch.arange(6).repeat_interleave(3),6,condition)
    tr=SimpleNamespace(step_i=0,n_conditions=2,train_crystals=torch.arange(6),condition=condition,
                       index=index,cfg=SimpleNamespace(seed=7,max_obs_per_batch=100))
    sizes=[]
    for step in range(6):
        tr.step_i=step
        k,rows=train._batch(tr)
        assert k==step%2
        crystals=rows//3
        assert torch.all(condition[crystals]==k)
        sizes.append(len(rows))
    assert sizes==[6,12,6,12,6,12]


@pytest.mark.parametrize("merge_friedel", [True, False])
def test_joint_train_and_checkpoint_readout(tmp_path,monkeypatch,merge_friedel):
    prep,_,*_=_synthetic(n_hkl=20,n_xtal=40,per=10,n_cond=2)
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1','d_min':0.9,'gain':1.0,'radii':[7,9,11],'merge_friedel':merge_friedel},['dark','light'],0.1)
    monkeypatch.setattr(P,'load_prep',lambda *a,**k: bundle)
    for key,value in dict(d_model=16,n_layers=2,inner_points=32,k_coarse=15,m_fine=65).items():
        monkeypatch.setitem(P.DEFAULTS,key,value)
    run=tmp_path/'run'
    train.main(['x.stream','-o',str(run),'--device','cpu','--steps','4','--seed','17'])
    summary=json.loads((run/'training_summary.json').read_text())
    assert summary['updates_per_state']==[2,2]
    ck=torch.load(run/'best.pt',weights_only=False)
    assert ck['scatter']['family']=='mixture' and ck['scatter']['n_components']==4
    assert ck['likelihood']['bg_mean_offset']==0.1
    assert ck['data']['training']['config']['seed']==17
    assert (run/'last.pt').exists()
    assert len(list(run.glob('*.mtz')))==6
    exported=[]
    original=P.write_mtz
    def capture(path,hkl,F,F_sd,**kwargs):
        exported.append((path,F.detach().cpu().numpy().copy(),F_sd.detach().cpu().numpy().copy(),kwargs['keep']))
        return original(path,hkl,F,F_sd,**kwargs)
    monkeypatch.setattr(P,'write_mtz',capture)
    merge_nn.main([str(run/'best.pt'),'x.stream','-o',str(tmp_path/'again.mtz'),'--device','cpu'])
    assert len(exported)==6
    import gemmi
    for path,F,sd,keep in exported:
        mtz=gemmi.read_mtz_file(str(path))
        labels = ['F-obs'] if merge_friedel else ['F-obs(+)', 'F-obs(-)']
        assert set(labels) <= set(mtz.column_labels())
        trained=gemmi.read_mtz_file(str(run/path.name.replace('again','merged')))
        a=np.concatenate([np.asarray(trained.column_with_label(label)) for label in labels])
        b=np.concatenate([np.asarray(mtz.column_with_label(label)) for label in labels])
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        a,b=a[np.isfinite(a)],b[np.isfinite(b)]
        assert np.corrcoef(a,b)[0,1] > 0.999
        assert np.median(b/a) == pytest.approx(1.0,rel=0.01)
        # Sorting survives ASU reordering and checks the actual per-merge sigma values.
        sigmas=np.concatenate([np.asarray(mtz.column_with_label(c))
                               for c in mtz.column_labels() if c.upper().startswith('SIG')])
        np.testing.assert_array_equal(np.sort(sigmas[np.isfinite(sigmas)]),np.sort(sd[keep]))
    assert not list(tmp_path.rglob('*pooled*'))
    assert not list(tmp_path.rglob('*.calibration.json'))


def test_readout_grid_rebuilds_nodes_and_interpolation():
    from torchsx.nn.posterior_target import PosteriorMergeTarget
    target=PosteriorMergeTarget(torch.ones(4),k_coarse=7,m_fine=17,n_sigma=6)
    target.set_grid(15,65,24)
    reference=PosteriorMergeTarget(torch.ones(4),k_coarse=15,m_fine=65,n_sigma=24)
    for key in ('u_coarse','u_fine','_lo','_hi','_t'):
        assert torch.equal(getattr(target,key),getattr(reference,key)),key
    assert target.u_coarse.numel()==15 and target.u_fine.numel()==65
