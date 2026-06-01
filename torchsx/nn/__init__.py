"""Neural-network-facing utilities for torchsx (padding, batching, scaling loss)."""

from torchsx.nn.batch import GroupedBatch, StreamBatcher
from torchsx.nn.consistency_loss import ScaleConsistencyNLL
from torchsx.nn.features import (
    DEFAULT_SCALE_FEATURES,
    TARGET_FIELDS,
    ReflectionBatch,
    ReflectionDataset,
)
from torchsx.nn.pad import group_offsets_from_counts, pack, unpack

__all__ = [
    "group_offsets_from_counts", "pack", "unpack",
    "GroupedBatch", "StreamBatcher",
    "ScaleConsistencyNLL",
    "ReflectionDataset", "ReflectionBatch",
    "DEFAULT_SCALE_FEATURES", "TARGET_FIELDS",
]
