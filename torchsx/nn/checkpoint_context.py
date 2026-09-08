"""Exact feature/target context for the versioned anisotropy experiment."""
import torch

TARGET_BUFFERS = (
    'ema_I0', 'ema_den', '_den_seeded', 'ema_log_sigma', '_sigma_seeded',
    'wilson_sigma', 'is_centric', 'wilson_phi', 'wilson_coef',
)


def capture_context(trainer):
    return {
        'schema_version': 1,
        'hkl': trainer.prep.hkl_valid.detach().cpu().clone(),
        'feature_reference': trainer.scale.features._I0_ref.detach().cpu().clone(),
        'targets': [{name: getattr(t, name).detach().cpu().clone()
                     for name in TARGET_BUFFERS if hasattr(t, name)}
                    for t in trainer.targets],
        'probes': capture_probes(trainer),
    }


@torch.no_grad()
def capture_probes(trainer):
    probes = []
    saved_condition = trainer._cond
    trainer.scale.eval()
    for condition in range(trainer.n_conditions):
        crystals = trainer.index.nonempty
        crystals = crystals[trainer.condition[crystals] == condition][:4]
        for _, rows in trainer.index.batches(100000, crystals=crystals, shuffle=False):
            meta, *_ = trainer._batch(rows.to(trainer.device))
            pred = trainer.scale.prime(meta)
            probes.append({'rows': rows.cpu().clone(), 'prediction': {
                key: value.detach().cpu().clone() for key, value in vars(pred).items()
                if torch.is_tensor(value)}})
            trainer.scale.clear()
    trainer._cond = saved_condition
    return probes


@torch.no_grad()
def verify_probes(trainer, context):
    errors = {}
    for probe in context['probes']:
        meta, *_ = trainer._batch(probe['rows'].to(trainer.device))
        pred = trainer.scale.prime(meta)
        for name, expected in probe['prediction'].items():
            actual = getattr(pred, name).cpu()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            errors[name] = max(errors.get(name, 0.), float((actual - expected).abs().max()))
        trainer.scale.clear()
    return errors


@torch.no_grad()
def restore_context(trainer, context):
    if context.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint context schema')
    if not torch.equal(context['hkl'].cpu(), trainer.prep.hkl_valid.cpu()):
        raise ValueError('Checkpoint context HKL order differs from prepared data')
    if len(context['targets']) != len(trainer.targets):
        raise ValueError('Checkpoint context condition count differs')
    # Standardization is already restored by load_state_dict. Recomputing it
    # here would no longer reproduce the exact evaluated network.
    ref = context['feature_reference']
    if ref.shape != trainer.scale.features._I0_ref.shape or not torch.isfinite(ref).all():
        raise ValueError('Invalid checkpoint feature reference')
    trainer.scale.features._I0_ref.copy_(ref)
    for target, buffers in zip(trainer.targets, context['targets']):
        for name, value in buffers.items():
            if name not in TARGET_BUFFERS:
                raise ValueError(f'Unexpected target buffer: {name}')
            current = getattr(target, name)
            if current.shape != value.shape:
                if name in ('wilson_phi', 'wilson_coef'):
                    # Basis-mode buffers are empty until a basis is installed.
                    setattr(target, name, value.to(current.device).clone())
                    continue
                raise ValueError(f'Checkpoint context buffer {name} has shape '
                                 f'{tuple(value.shape)}, expected {tuple(current.shape)}')
            current.copy_(value)
    trainer.scale.clear()


@torch.no_grad()
def load_checkpoint_context(trainer, checkpoint, mode='exact'):
    if mode == 'exact':
        if 'inference_context' not in checkpoint:
            raise ValueError('Legacy checkpoint lacks exact inference context; '
                             'use an explicitly labelled legacy/ema diagnostic')
        restore_context(trainer, checkpoint['inference_context'])
    elif mode in ('legacy', 'ema'):
        ema = checkpoint.get('ema_I0')
        if ema is None:
            raise ValueError('Checkpoint lacks running intensities')
        for k, t in enumerate(trainer.targets):
            t.ema_I0.copy_(ema if ema.ndim == 1 else ema[k])
        if mode == 'ema':
            trainer._refresh_feature_reference()
    else:
        raise ValueError(mode)
