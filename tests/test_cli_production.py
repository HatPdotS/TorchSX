"""Production inputs, fixed-update training, checkpoint replay and raw products."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import warnings

import numpy as np
import pytest
import torch

from torchsx.cli import _pipeline as P, train, merge_nn
from tests.test_amortized_training import _synthetic
from tests.test_cli_pipeline import _stack, _DataArgs
from tests.test_measurement_sources import stream_file


@pytest.mark.parametrize('module', [train, merge_nn])
def test_split_inputs(module):
    parser = module._build_parser()
    prefix = [] if module is train else ['model.pt']
    a = parser.parse_args(prefix + ['--split','dark=dark.stream','--split','light=light.stream','-o','run'])
    assert P._splits_from_args(a) == [('dark',Path('dark.stream')),('light',Path('light.stream'))]
    a = parser.parse_args(prefix + ['combined.stream','--split-file','custom.split','-o','run'])
    assert a.split_file == Path('custom.split') and a.unassigned is None
    assert parser.parse_args(prefix + ['x.stream','--unassigned','drop','-o','run']).unassigned == 'drop'


def test_inputs_of_one_name_form_one_dataset(tmp_path, monkeypatch):
    def parse(*argv):
        return P._splits_from_args(train._build_parser().parse_args([*argv, '-o', 'run']))
    pooled = parse('--split','dark=a.stream','--split','light=c.stream','--split','dark=b.stream')
    assert pooled == [('dark',Path('a.stream')),('light',Path('c.stream')),('dark',Path('b.stream'))]
    assert P._dataset_names(pooled) == ['dark','light']
    bare = parse('a.stream','b.stream')
    assert bare == [('',Path('a.stream')),('',Path('b.stream'))] and P._dataset_names(bare) == ['']
    assert parse('a.stream','b.stream','--split-file','custom.split') == bare
    for bad in (('--split','dark=a.stream','b.stream'),
                ('--split','dark=a.stream','--split','light=b.stream','--split-file','custom.split'),
                ('--split',f'{P.UNASSIGNED}=a.stream')):
        with pytest.raises(SystemExit):
            parse(*bad)
    # a pattern is expanded here, in sorted order; a file of that very name is taken as it is
    monkeypatch.chdir(tmp_path)
    for name in ('run2.stream','run1.stream','other.sfx','odd[1].stream'):
        (tmp_path/name).touch()
    assert parse('--split','dark=run*.stream','--split','light=other.sfx','--split','dark=odd[1].stream') == [
        ('dark',Path('run1.stream')),('dark',Path('run2.stream')),('light',Path('other.sfx')),
        ('dark',Path('odd[1].stream'))]
    with pytest.raises(SystemExit, match='matches no file'):
        parse('--split','dark=none*.stream')


def test_updates_follow_the_share_of_the_patterns_of_no_dataset():
    def states(n_conditions, share, steps=400):
        tr = SimpleNamespace(n_conditions=n_conditions, unassigned_share=share)
        return [train._state_of_step(tr, i) for i in range(steps)]
    # every pattern in a dataset: the states take turns, one draw per round
    assert states(2, None, 6) == [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
    for n, share in ((3, 0.25), (3, 0.9), (2, 0.5), (4, 0.01), (2, 0.0)):
        got = states(n, share)
        per = np.bincount([k for k, _ in got], minlength=n)
        assert per[-1] == int(400 * share)                      # the last state, by its share
        assert per[:-1].max() - per[:-1].min() <= 1             # the datasets, evenly
        assert len({draw for _, draw in got}) == 400            # no batch drawn twice
        # spread over the run rather than bunched
        assert abs(sum(k == n - 1 for k, _ in got[:200]) - 200 * share) <= 1

    condition = torch.tensor([0, 0, 1, 2, 2, 2, 2, 0])
    tr = SimpleNamespace(n_conditions=3, condition=condition, train_crystals=torch.arange(7))
    assert train.unassigned_share(tr, ['dark', 'light', P.UNASSIGNED]) == pytest.approx(4 / 7)
    assert train.unassigned_share(tr, ['dark', 'light', 'late']) is None
    assert train.unassigned_share(tr, []) is None


def test_patterns_of_no_dataset_train_and_are_not_merged(tmp_path, monkeypatch, capsys):
    prep,_,*_=_synthetic(n_hkl=20,n_xtal=40,per=10,n_cond=2)
    # every fourth crystal is a pattern of no dataset, and every eighth is in both
    i=torch.arange(prep.obs_info.n_crystals)
    free=i%4==3
    member=torch.stack([(i%2==0)|(i%8==1), (i%2==1)&~free, free],1)
    P.assign_states(prep.obs_info,member)
    names=['dark','light',P.UNASSIGNED]
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1','d_min':0.9,'gain':1.0,'radii':[7,9,11]},names,0.1)
    monkeypatch.setattr(P,'load_prep',lambda *a,**k: bundle)
    for key,value in dict(d_model=16,inner_points=32,k_coarse=15,m_fine=65).items():
        monkeypatch.setitem(P.DEFAULTS,key,value)
    run=tmp_path/'run'
    train.main(['x.stream','-o',str(run),'--device','cpu','--steps','8','--seed','17'])
    updates=json.loads((run/'training_summary.json').read_text())['updates_per_state']
    assert sum(updates)==8 and updates[2]==2 and updates[0]==updates[1]==3
    ck=torch.load(run/'best.pt',weights_only=False)
    assert ck['splits']==names and len(ck['wilson_surface']['coef'])==3
    wanted={f'merged_{n}{half}.mtz' for n in ('dark','light') for half in ('','_half1','_half2')}
    assert {p.name for p in run.glob('*.mtz')}==wanted
    assert set(json.loads((run/'merged.metrics.json').read_text()))=={'dark','light'}
    out=capsys.readouterr().out
    # the crystals of both states are merged in each, held once
    assert 'merge [dark]: 25 crystals' in out and 'merge [light]: 10 crystals' in out
    merge_nn.main([str(run/'best.pt'),'x.stream','-o',str(tmp_path/'again.mtz'),'--device','cpu'])
    assert {p.name for p in tmp_path.glob('*.mtz')}=={n.replace('merged','again') for n in wanted}


def test_research_options_parse_but_stay_out_of_help():
    parser = train._build_parser()
    hidden = {'--noise-lr-mult': '10', '--context-folds': '3', '--d-model': '64',
              '--eval-every': '50', '--heldout-stream': 'h.stream', '--outlier-frac': '0.01',
              '--ctx-layers': '4', '--query-layers': '0', '--tether-weight': '2'}
    flags = ['--refine-geometry']
    text = parser.format_help()
    assert not [o for o in hidden if o in text]
    assert not [o for o in flags if o in text]
    a = parser.parse_args(['x.stream', '-o', 'run', *(t for o, v in hidden.items() for t in (o, v)), *flags])
    assert (a.noise_lr_mult, a.context_folds, a.d_model, a.eval_every, a.heldout_stream,
            a.outlier_frac) == (10.0, 3, 64, 50, Path('h.stream'), 0.01)
    assert (a.ctx_layers, a.query_layers, a.refine_geometry) == (4, 0, True)
    assert a.tether_weight == 2.0 and a.tether is None
    assert parser.parse_args(['x.stream', '-o', 'run', '--tether', 'gauge']).tether == ('gauge',)
    assert parser.parse_args(['x.stream', '-o', 'run', '--tether', 'none']).tether == ()
    assert parser.parse_args(['x.stream', '-o', 'run']).pseudo_translation == 'auto'
    assert parser.parse_args(['x.stream', '-o', 'run', '--no-refine-geometry']).refine_geometry is False
    assert parser.parse_args(['x.stream', '-o', 'run']).steps == 400


@pytest.mark.parametrize('option,value', [('--miss-frac', '1'), ('--outlier-frac', '-0.1'), ('--noise-terms', 'C'),
                                          ('--tether', 'gauge,wobble'), ('--tether-weight', '0')])
def test_out_of_range_training_options_fail_at_parsing(option, value, capsys):
    with pytest.raises(SystemExit):
        train._build_parser().parse_args(['x.stream', '-o', 'run', option, value])
    assert value in capsys.readouterr().err


@pytest.mark.parametrize('module', [train,merge_nn])
@pytest.mark.parametrize('option,value', [('--gain','garbage'),('--prior','none'),('--scatter-family','mixture'),('--calib-degree','9,9')])
def test_invalid_or_removed_options_fail_before_loading(module,option,value,capsys):
    prefix = [] if module is train else ['model.pt']
    with pytest.raises(SystemExit):
        module.main(prefix + ['missing.stream','-o','run',option,value])
    error = capsys.readouterr().err
    assert value in error
    if option == '--gain':
        for valid in ('auto','trust_input','calibrate','positive finite'):
            assert valid in error
    if option == '--prior':
        assert 'wilson' in error


def test_custom_split_matches_source_images(stream_file,tmp_path):
    labels = tmp_path/'custom.split'
    labels.write_text('/raw/a.h5 //5 dark\n/raw/b.h5 //8 light\n')
    args = _DataArgs(stream=[stream_file],split_file=labels,int_radius=None,gain='trust_input')
    bundle=P.load_prep(args,verbose=False)
    assert bundle.split_names == ['dark','light']
    assert set(bundle.prep.obs_info.delay_bin_idx.tolist()) == {0,1}


def test_cache_moves_with_inputs_and_rejects_changed_content(stream_file,tmp_path,monkeypatch):
    cache=tmp_path/'prepared'
    args=_DataArgs(stream=[stream_file],int_radius=None,gain=None,cache_prep=cache)
    built=P.load_prep(args,verbose=False)
    moved=tmp_path/'moved';moved.mkdir()
    source=moved/stream_file.name;shutil.copyfile(stream_file,source)
    relocated=moved/'prepared';shutil.copytree(cache,relocated)
    args.stream,args.cache_prep=[source],relocated
    monkeypatch.setattr('torchsx.merging.prepare_store.StoreBuilder.add',lambda *a,**k: pytest.fail('cache hit reparsed observations'))
    loaded=P.load_prep(args,recipe_defaults=built.recipe,verbose=False)
    assert torch.equal(loaded.prep.materialize().I_obs,built.prep.materialize().I_obs)
    source.write_text(source.read_text()+'\n# changed input\n')
    with pytest.raises(SystemExit,match='different settings'):
        P.load_prep(args,verbose=False)


def test_defaults_and_old_checkpoint_likelihood(tmp_path):
    prep,params,*_=_synthetic(n_hkl=20,n_xtal=40,per=10)
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1'},[],0.125)
    args=train._build_parser().parse_args(['x.stream','-o',str(tmp_path),'--device','cpu'])
    host,_,scale,obj,S=P.build_stack(bundle,args,verbose=False)
    assert host is prep
    assert scale.heads.n_components==4
    assert obj.likelihood.bg_mean_offset==0.125
    assert S['prior']=='wilson'
    oldscale,oldobj,tr=_stack(prep,params,tmp_path)
    try:
        ck=torch.load(tr.save_best(),weights_only=False)
        _,_,rebuilt,oldlik,_=P.build_stack(bundle,args,ckpt=ck,verbose=False)
        rebuilt.load_state_dict(ck['state_dict'])
        assert oldlik.likelihood.bg_mean_offset==0.0
    finally:
        tr.close()


@pytest.mark.parametrize('saved,requested,expected', [
    ('wilson',None,'wilson'), ('wilson','wilson','wilson'),
    ('none',None,SystemExit), ('none','wilson',SystemExit),
])
def test_checkpoint_prior_wilson_and_removed_flat_prior(tmp_path,saved,requested,expected):
    prep,_,*_=_synthetic(n_hkl=20,n_xtal=40,per=10)
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1'},[],0.125)
    argv=['model.pt','x.stream','-o',str(tmp_path/'merged.mtz'),'--device','cpu']
    if requested is not None:
        argv+=['--prior',requested]
    args=merge_nn._build_parser().parse_args(argv)
    ck=_production_checkpoint(bundle,tmp_path)
    ck['posterior_grid']={'prior':saved}
    if expected is SystemExit:
        # A flat-prior checkpoint (#24) carries no Wilson surface: no override rescues it.
        with pytest.raises(SystemExit, match='--prior none'):
            P.build_stack(bundle,args,ckpt=ck,verbose=False)
        return
    _,_,_,_,settings=P.build_stack(bundle,args,ckpt=ck,verbose=False)
    assert settings['prior']==expected


def _production_checkpoint(bundle,tmp_path):
    """A checkpoint of the default model, written by the trainer."""
    from tests.test_cli_pipeline import _trainer_from_stack
    targs=train._build_parser().parse_args(['x.stream','-o',str(tmp_path),'--device','cpu'])
    tr=_trainer_from_stack(*P.build_stack(bundle,targs,verbose=False))
    try:
        return torch.load(tr.save_best(tmp_path/'model.pt'),weights_only=False)
    finally:
        tr.close()


def test_batches_are_balanced_and_allow_small_states():
    from torchsx.nn.train_amortized import CrystalIndex
    condition=torch.tensor([0,0,1,1,1,1])
    index=CrystalIndex(torch.arange(6).repeat_interleave(3),6,condition)
    tr=SimpleNamespace(step_i=0,n_conditions=2,train_crystals=torch.arange(6),condition=condition,
                       index=index,cfg=SimpleNamespace(seed=7,max_obs_per_batch=100))
    sizes=[]
    with pytest.warns(UserWarning, match='using all') as caught:
        for step in range(6):
            tr.step_i=step
            k,rows=train._batch(tr)
            assert k==step%2
            crystals=rows//3
            assert torch.all(condition[crystals]==k)
            sizes.append(len(rows))
    assert len(caught)==2  # Once per state, rather than once per update.
    assert sizes==[6,12,6,12,6,12]


@pytest.mark.parametrize('requested', [6, 9, 100])
def test_small_batch_respects_holdout_and_original_redundancy(requested):
    from torchsx.nn.train_amortized import CrystalIndex
    condition=torch.zeros(3, dtype=torch.long)
    index=CrystalIndex(torch.arange(3).repeat_interleave(3),3,condition)
    tr=SimpleNamespace(step_i=0,n_conditions=1,train_crystals=torch.tensor([0,1]),
                       condition=condition,index=index,prep=SimpleNamespace(n_valid=1),
                       cfg=SimpleNamespace(seed=7,max_obs_per_batch=min(requested,9),
                                           target_redundancy=requested))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, rows=train._batch(tr)
    assert set(rows.tolist())==set(range(6))  # Crystal 2 is held out.
    assert len(caught)==int(requested>6)
    if caught:
        assert f'requested batch size of {requested:,}' in str(caught[0].message)


@pytest.mark.parametrize("merge_friedel", [True, False])
def test_joint_train_and_checkpoint_readout(tmp_path,monkeypatch,merge_friedel):
    prior='wilson'
    prep,_,*_=_synthetic(n_hkl=20,n_xtal=40,per=10,n_cond=2)
    bundle=P.PrepBundle(prep,{'spacegroup':'P 1','d_min':0.9,'gain':1.0,'radii':[7,9,11],'merge_friedel':merge_friedel},['dark','light'],0.1)
    monkeypatch.setattr(P,'load_prep',lambda *a,**k: bundle)
    for key,value in dict(d_model=16,inner_points=32,k_coarse=15,m_fine=65).items():
        monkeypatch.setitem(P.DEFAULTS,key,value)
    run=tmp_path/'run'
    train.main(['x.stream','-o',str(run),'--device','cpu','--steps','4','--seed','17',
                '--prior',prior])
    summary=json.loads((run/'training_summary.json').read_text())
    assert summary['updates_per_state']==[2,2]
    res=summary['resources']
    assert list(res['phases'])==['prepare','train','readout']
    assert all(p['seconds']>=0 and 'gpu_peak_allocated_gib' not in p for p in res['phases'].values())
    assert res['updates']['n']==4 and res['updates']['median_s']<=res['updates']['max_s']
    ck=torch.load(run/'best.pt',weights_only=False)
    assert ck['scatter']['n_components']==4
    assert ck['likelihood']['bg_mean_offset']==0.1
    assert ck['data']['training']['config']['seed']==17
    assert (ck['wilson_surface'] is not None and len(ck['wilson_surface']['coef'])==2)
    assert ck['posterior_grid']['prior']==prior
    assert ck['data']['training']['config']['posterior_prior']==prior
    assert ck['data']['training']['readout']['prior']==prior
    assert all(r['prior']==prior for r in json.loads((run/'merged.metrics.json').read_text()).values())
    assert (run/'last.pt').exists()
    assert len(list(run.glob('*.mtz')))==6
    exported=[]
    original=P.write_mtz
    def capture(path,hkl,F,F_sd,**kwargs):
        exported.append((path,F.detach().cpu().numpy().copy(),F_sd.detach().cpu().numpy().copy(),kwargs['keep']))
        return original(path,hkl,F,F_sd,**kwargs)
    monkeypatch.setattr(P,'write_mtz',capture)
    merge_nn.main([str(run/'best.pt'),'x.stream','-o',str(tmp_path/'again.mtz'),'--device','cpu'])
    assert all(r['prior']==prior for r in json.loads((tmp_path/'again.metrics.json').read_text()).values())
    assert list(json.loads((tmp_path/'again.resources.json').read_text())['phases'])==['prepare','readout']
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
        # Separate readout forms every per-reflection input from the same data
        # and takes the trained Wilson surface: it reproduces training's merge.
        np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(np.sort(b[np.isfinite(b)]), np.sort(F[keep]))
        # Sorting survives ASU reordering and checks the actual per-merge sigma values.
        sigmas=np.concatenate([np.asarray(mtz.column_with_label(c))
                               for c in mtz.column_labels() if c.upper().startswith('SIG')])
        np.testing.assert_array_equal(np.sort(sigmas[np.isfinite(sigmas)]),np.sort(sd[keep]))
    assert not list(tmp_path.rglob('*pooled*'))
    assert not list(tmp_path.rglob('*.calibration.json'))


def test_separate_merge_exports_new_bins(tmp_path, monkeypatch):
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=24, per=10, n_cond=2)
    _, _, trainer = _stack(prep, params, tmp_path)
    trainer.split_names = ['dark', 'light']
    try:
        checkpoint = trainer.save_best(tmp_path / 'model.pt')
    finally:
        trainer.close()

    current, *_ = _synthetic(n_hkl=20, n_xtal=24, per=10, n_cond=3, seed=9)
    names = ['early', 'middle', 'late']
    bundle = P.PrepBundle(current, {'spacegroup': 'P 1'}, names, 0.1)
    monkeypatch.setattr(P, 'load_prep', lambda *a, **kw: bundle)
    monkeypatch.setitem(P.DEFAULTS, 'k_coarse', 15)
    monkeypatch.setitem(P.DEFAULTS, 'm_fine', 65)
    merge_nn.main([str(checkpoint), 'new.stream', '-o', str(tmp_path / 'rebinned.mtz'),
                   '--device', 'cpu'])

    metrics = json.loads((tmp_path / 'rebinned.metrics.json').read_text())
    assert set(metrics) == {'early', 'middle', 'late'}
    assert {p.name for p in tmp_path.glob('*.mtz')} == {
        f'rebinned_{name}{half}.mtz'
        for name in names for half in ('', '_half1', '_half2')
    }


def test_readout_grid_rebuilds_nodes_and_interpolation():
    from torchsx.nn.posterior_target import PosteriorMergeTarget
    target=PosteriorMergeTarget(torch.ones(4),k_coarse=7,m_fine=17,n_sigma=6)
    target.set_grid(15,65,24)
    reference=PosteriorMergeTarget(torch.ones(4),k_coarse=15,m_fine=65,n_sigma=24)
    for key in ('u_coarse','u_fine','_lo','_hi','_t'):
        assert torch.equal(getattr(target,key),getattr(reference,key)),key
    assert target.u_coarse.numel()==15 and target.u_fine.numel()==65


def test_split_half_stats_report_shells_and_quintiles():
    from torchsx.cli.merge_nn import _split_half_stats

    g = torch.Generator().manual_seed(0)
    n = 500
    truth = torch.rand(n, generator=g).double() * 100
    a = {"seen": torch.ones(n, dtype=torch.bool), "I0": truth + torch.randn(n, generator=g).double()}
    b = {"seen": torch.ones(n, dtype=torch.bool), "I0": truth + torch.randn(n, generator=g).double()}
    d = torch.linspace(1.5, 20.0, n)
    stats = _split_half_stats(a, b, d, rank=truth)
    assert len(stats["by_shell"]) == 10 and len(stats["by_quintile"]) == 5
    assert stats["by_shell"][0]["d_lo"] > stats["by_shell"][-1]["d_lo"]
    assert sum(q["n"] for q in stats["by_quintile"]) == n
    # the weakest quintile has the largest relative disagreement
    assert stats["by_quintile"][0]["r_split"] > stats["by_quintile"][-1]["r_split"]
    assert stats["cc_half"] > 0.99


def test_also_exports_the_requested_estimators(tmp_path, monkeypatch, capsys):
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=24, per=10, n_cond=1)
    _, _, trainer = _stack(prep, params, tmp_path)
    try:
        checkpoint = trainer.save_best(tmp_path / 'model.pt')
    finally:
        trainer.close()
    bundle = P.PrepBundle(prep, {'spacegroup': 'P 1'}, [''], 0.1)
    monkeypatch.setattr(P, 'load_prep', lambda *a, **kw: bundle)
    monkeypatch.setitem(P.DEFAULTS, 'k_coarse', 15)
    monkeypatch.setitem(P.DEFAULTS, 'm_fine', 65)
    merge_nn.main([str(checkpoint), 'x.stream', '-o', str(tmp_path / 'm.mtz'), '--device', 'cpu',
                   '--also', 'posterior', 'pooled', 'flat', 'wls'])
    assert {p.name for p in tmp_path.glob('m*.mtz')} == {
        'm.mtz', 'm_half1.mtz', 'm_half2.mtz', 'm_posterior.mtz', 'm_pooled.mtz', 'm_flat.mtz'}
    rec = json.loads((tmp_path / 'm.metrics.json').read_text())['']
    assert rec['estimator'] == 'wls'                      # the default product; not exported twice
    assert rec['products'] == ['wls', 'posterior', 'pooled', 'flat']
    assert 'n' in rec['top_edge'] and 'split_half_wls' in rec
    tables = {name: np.genfromtxt(tmp_path / f'm_{name}.tsv', names=True, delimiter='\t')
              for name in ('posterior', 'pooled', 'flat', 'wls')}
    assert tables['pooled'].dtype.names == tables['wls'].dtype.names == ('h', 'k', 'l', 'd_A', 'I', 'sigI', 'n_meas')
    assert tables['posterior'].dtype.names == ('h', 'k', 'l', 'd_A', 'F', 'sigF', 'I0', 'sigI0', 'n_meas')
    # every observation is counted once, by the moment and the posterior readouts alike
    assert tables['pooled']['n_meas'].sum() == prep.n_obs_total == tables['posterior']['n_meas'].sum()
    assert np.isfinite(tables['flat']['F']).all() and (tables['flat']['sigF'] > 0).all()
    assert 'also        pooled' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        merge_nn._build_parser().parse_args(['model.pt', 'x.stream', '-o', 'm.mtz', '--also', 'median'])


def test_top_edge_reports_the_multiplicity_of_edge_piled_reflections():
    seen = torch.tensor([True, True, True, True, False])
    res = {'seen': seen, 'F': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
           'F_sd': torch.full((5,), 0.1),
           'edge_high': torch.tensor([0.0, 0.5, 0.02, 0.001, 0.9]),
           'n_meas': torch.tensor([100, 2, 4, 50, 1])}
    top = P.top_edge(res, np.arange(15).reshape(5, 3))
    assert top['n'] == 2                                  # the unseen reflection is not counted
    assert [r['hkl'] for r in top['worst']] == [[3, 4, 5], [6, 7, 8]]
    assert [r['n_meas'] for r in top['worst']] == [2, 4]
    assert top['n_meas_median'] == 2.0 and top['n_meas_median_all'] == 4.0
    assert P.top_edge({**res, 'edge_high': torch.zeros(5)})['n'] == 0
    assert P.top_edge({'seen': seen, 'F': res['F'], 'F_sd': res['F_sd']})['n'] == 0



def test_estimator_selects_the_product(tmp_path, monkeypatch, capsys):
    """wls by default: French-Wilson amplitudes of the inverse-variance merge and its
    halves. posterior: the posterior amplitudes, with posterior halves. Both record the plain
    and the flat CC1/2 of the half files."""
    import gemmi
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=24, per=10, n_cond=1)
    _, _, trainer = _stack(prep, params, tmp_path)
    try:
        checkpoint = trainer.save_best(tmp_path / 'model.pt')
    finally:
        trainer.close()
    bundle = P.PrepBundle(prep, {'spacegroup': 'P 1'}, [''], 0.1)
    monkeypatch.setattr(P, 'load_prep', lambda *a, **kw: bundle)
    monkeypatch.setitem(P.DEFAULTS, 'k_coarse', 15)
    monkeypatch.setitem(P.DEFAULTS, 'm_fine', 65)
    base = [str(checkpoint), 'x.stream', '--device', 'cpu']
    merge_nn.main(base + ['-o', str(tmp_path / 'w.mtz'), '--also', 'wls'])
    merge_nn.main(base + ['-o', str(tmp_path / 'p.mtz'), '--estimator', 'posterior'])
    out = capsys.readouterr().out
    assert 'half-datasets  wls' in out and 'half-datasets  posterior' in out
    assert out.count('flat CC1/2') == 2
    w = json.loads((tmp_path / 'w.metrics.json').read_text())['']
    p = json.loads((tmp_path / 'p.metrics.json').read_text())['']
    assert (w['estimator'], w['products']) == ('wls', ['wls'])
    assert 'split_half_wls' in w and 'split_half_pooled' not in w and 'sigF_collapsed' not in w
    assert (p['estimator'], p['products']) == ('posterior', ['posterior'])
    assert {'split_half_posterior', 'sigF_collapsed'} <= set(p) and 'split_half_pooled' not in p
    from torchsx.merge_stats import load_merge, split_half_stats
    for stem, rec in (('w', w), ('p', p)):
        files = rec['split_half_files']
        direct = split_half_stats(load_merge(tmp_path / f'{stem}_half1.mtz'), load_merge(tmp_path / f'{stem}_half2.mtz'))
        assert files['cc_half_flat'] == pytest.approx(direct['headline']['cc_half_flat'], nan_ok=True)
        assert len(files['by_shell']) == len(direct['shells'])
    for stem in ('w', 'w_half1', 'w_half2', 'p', 'p_half1', 'p_half2'):
        mtz = gemmi.read_mtz_file(str(tmp_path / f'{stem}.mtz'))
        F = np.asarray(mtz.column_with_label('F-obs'))
        assert np.isfinite(F).all() and (F > 0).all()
    # every wls intensity gets a French-Wilson amplitude, and the amplitudes follow the intensities
    table = np.genfromtxt(tmp_path / 'w_wls.tsv', names=True, delimiter='\t')
    F = np.asarray(gemmi.read_mtz_file(str(tmp_path / 'w.mtz')).column_with_label('F-obs'))
    assert len(F) == len(table)
    assert np.corrcoef(np.sort(F), np.sort(np.sqrt(np.clip(table['I'], 0, None))))[0, 1] > 0.9
