"""Neural-network-facing utilities for torchsx (padding, batching, scaling loss).

The amortised scale model lives here: :mod:`~torchsx.nn.pooling` (the
leave-one-out primitives everything else is built on), :mod:`~torchsx.nn.deepsets`
(the abismal encoder port), :mod:`~torchsx.nn.heads`,
:mod:`~torchsx.nn.obs_features` (the feature contract),
:mod:`~torchsx.nn.merge_target`, :mod:`~torchsx.nn.objective`,
:mod:`~torchsx.nn.metrics` and :mod:`~torchsx.nn.train_amortized`.  The model
itself is :class:`torchsx.merging.scale.AmortizedScale`, because it must be a
``ScaleModel`` for the rest of the merging pipeline to consume it.
"""

from torchsx.nn.deepsets import DeepSetsEncoder, ResidualFF, ScaleEncoder
from torchsx.nn.heads import ScaleHeads, ScalePrediction
from torchsx.nn.merge_target import LOOInBatchTarget, MergeTargetOut
from torchsx.nn.metrics import (
    MetricLogger,
    calibration_metrics,
    condition_auc,
    hkl_permutation_sensitivity,
    scale_metrics,
)
from torchsx.nn.objective import AgreementNLL, MarginalNLL
from torchsx.nn.obs_features import ObsFeatureSpec
from torchsx.nn.pad import group_offsets_from_counts, pack, unpack
from torchsx.nn.pooling import leave_one_out_mean, loo_weighted_mean
from torchsx.nn.train_amortized import AmortizedTrainer, CrystalIndex, TrainConfig

__all__ = [
    "group_offsets_from_counts", "pack", "unpack",
    # amortised scale model
    "leave_one_out_mean", "loo_weighted_mean",
    "ScaleEncoder", "ResidualFF", "DeepSetsEncoder",
    "ScaleHeads", "ScalePrediction",
    "ObsFeatureSpec",
    "LOOInBatchTarget", "MergeTargetOut",
    "MarginalNLL", "AgreementNLL",
    "MetricLogger", "calibration_metrics", "scale_metrics",
    "condition_auc", "hkl_permutation_sensitivity",
    "AmortizedTrainer", "TrainConfig", "CrystalIndex",
]
