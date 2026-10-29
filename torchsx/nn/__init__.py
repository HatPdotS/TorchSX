"""The amortised scale model's network and its training.

:mod:`~torchsx.nn.obs_features` builds the per-observation features and
enforces which of them the scale of an observation may see;
:mod:`~torchsx.nn.deepsets` encodes each crystal's observations;
:mod:`~torchsx.nn.heads` turns the embedding into ``log K`` and the scatter;
:mod:`~torchsx.nn.objective` and :mod:`~torchsx.nn.posterior_target` give the
marginal likelihood the network is trained on; :mod:`~torchsx.nn.train_amortized`
runs the training.  The model itself is
:class:`torchsx.merging.scale.AmortizedScale`.
"""

from torchsx.nn.deepsets import DeepSetsEncoder, ResidualFF, ScaleEncoder
from torchsx.nn.fold_transformer import FoldTransformerEncoder
from torchsx.nn.heads import ScaleHeads, ScalePrediction
from torchsx.nn.merge_target import MergeTargetOut
from torchsx.nn.metrics import (
    MetricLogger,
    calibration_metrics,
    condition_auc,
    scale_metrics,
)
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.obs_features import ObsFeatureSpec
from torchsx.nn.pooling import leave_one_out_mean, loo_weighted_mean
from torchsx.nn.train_amortized import AmortizedTrainer, CrystalIndex, TrainConfig

__all__ = [
    # amortised scale model
    "leave_one_out_mean", "loo_weighted_mean",
    "ScaleEncoder", "ResidualFF", "DeepSetsEncoder", "FoldTransformerEncoder",
    "ScaleHeads", "ScalePrediction",
    "ObsFeatureSpec",
    "MergeTargetOut",
    "MarginalNLL",
    "MetricLogger", "calibration_metrics", "scale_metrics",
    "condition_auc",
    "AmortizedTrainer", "TrainConfig", "CrystalIndex",
]
