"""Staged refinement loop for the pixel-level forward model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from torchsx.deprecated.forward.forward_model import PixelForwardModel
from torchsx.deprecated.forward.likelihood import PixelLikelihood
from torchsx.deprecated.forward.batch import ROIBatchLoader


@dataclass
class RefinementStage:
    """Configuration for a single refinement stage.

    Parameters
    ----------
    name : str — human-readable stage name
    n_epochs : int — number of epochs (full passes over data)
    lr : float — learning rate
    frozen : list[str] — parameter group names to freeze
        Valid names: "orientation", "spectrum", "structure_factors",
        "crystal_scale", "crystal_bfactor", "relp_profile", "geometry"
    predict_every : int — re-predict reflections every N epochs (0 = never)
    """

    name: str
    n_epochs: int
    lr: float = 1e-3
    frozen: list[str] = field(default_factory=list)
    predict_every: int = 0


# Default staged refinement schedule
BOOTSTRAP = RefinementStage(
    name="bootstrap",
    n_epochs=5,
    lr=1e-3,
    frozen=["orientation", "spectrum", "geometry"],
)
SPECTRAL = RefinementStage(
    name="spectral",
    n_epochs=5,
    lr=5e-4,
    frozen=["orientation", "geometry"],
)
JOINT = RefinementStage(
    name="joint",
    n_epochs=10,
    lr=2e-4,
    frozen=[],
)
WEAK_SPOTS = RefinementStage(
    name="weak_spots",
    n_epochs=5,
    lr=2e-4,
    frozen=[],
    predict_every=3,
)

DEFAULT_STAGES = [BOOTSTRAP, SPECTRAL, JOINT, WEAK_SPOTS]


def _module_for_group(model: PixelForwardModel, name: str) -> nn.Module:
    """Resolve a parameter group name to its sub-module."""
    mapping = {
        "orientation": model.orientation,
        "spectrum": model.spectrum,
        "structure_factors": model.structure_factors,
        "crystal_scale": model.crystal_scale,
        "crystal_bfactor": model.crystal_bfactor,
        "relp_profile": model.relp_profile,
        "geometry": model.geometry,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown parameter group {name!r}. "
            f"Valid: {list(mapping.keys())}"
        )
    return mapping[name]


def _set_requires_grad(model: PixelForwardModel, frozen: list[str]):
    """Enable grad for all parameters, then freeze specified groups."""
    for p in model.parameters():
        p.requires_grad_(True)
    for name in frozen:
        mod = _module_for_group(model, name)
        for p in mod.parameters():
            p.requires_grad_(False)


def refine(
    model: PixelForwardModel,
    loader: ROIBatchLoader,
    stages: Sequence[RefinementStage] | None = None,
    likelihood: PixelLikelihood | None = None,
    verbose: bool = True,
) -> PixelForwardModel:
    """Run staged refinement of the pixel-level forward model.

    Parameters
    ----------
    model : PixelForwardModel — the model to refine
    loader : ROIBatchLoader — data loader
    stages : refinement stages (default: BOOTSTRAP → SPECTRAL → JOINT → WEAK_SPOTS)
    likelihood : pixel likelihood (default: Gaussian)
    verbose : print progress

    Returns
    -------
    model : the refined model (same object, modified in-place)
    """
    if stages is None:
        stages = DEFAULT_STAGES
    if likelihood is None:
        likelihood = PixelLikelihood(mode="gaussian")

    device = next(model.parameters()).device

    for stage in stages:
        if verbose:
            print(f"=== Stage: {stage.name} ({stage.n_epochs} epochs, lr={stage.lr}) ===")
            if stage.frozen:
                print(f"    Frozen: {stage.frozen}")

        # Freeze/unfreeze parameters
        _set_requires_grad(model, stage.frozen)

        # Build optimizer with only trainable parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            if verbose:
                print("    No trainable parameters, skipping.")
            continue

        optimizer = torch.optim.Adam(trainable, lr=stage.lr)

        for epoch in range(stage.n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                n_pred = model(batch)
                loss = likelihood(n_pred, batch.patches.float(), batch.valid)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if verbose:
                print(f"    Epoch {epoch+1}/{stage.n_epochs}: loss = {avg_loss:.2f}")

    return model


def refine_sparse(
    model: PixelForwardModel,
    loader: "SparseBatchLoader",
    stages: Sequence[RefinementStage] | None = None,
    loss_fn: "CensoredPoissonLoss | None" = None,
    verbose: bool = True,
) -> PixelForwardModel:
    """Run staged refinement against sparse compressed data.

    Uses CensoredPoissonLoss instead of PixelLikelihood, and
    model.forward_sparse() instead of model.forward().

    Parameters
    ----------
    model : PixelForwardModel — the model to refine
    loader : SparseBatchLoader — sparse data loader
    stages : refinement stages (default: DEFAULT_STAGES)
    loss_fn : CensoredPoissonLoss (default: CensoredPoissonLoss())
    verbose : print progress

    Returns
    -------
    model : the refined model (same object, modified in-place)
    """
    from torchsx.deprecated.forward._censored_poisson import CensoredPoissonLoss

    if stages is None:
        stages = DEFAULT_STAGES
    if loss_fn is None:
        loss_fn = CensoredPoissonLoss()

    device = next(model.parameters()).device

    for stage in stages:
        if verbose:
            print(f"=== Stage: {stage.name} ({stage.n_epochs} epochs, lr={stage.lr}) ===")
            if stage.frozen:
                print(f"    Frozen: {stage.frozen}")

        _set_requires_grad(model, stage.frozen)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            if verbose:
                print("    No trainable parameters, skipping.")
            continue

        optimizer = torch.optim.Adam(trainable, lr=stage.lr)

        for epoch in range(stage.n_epochs):
            total_loss = 0.0
            total_t1 = 0.0
            total_t2 = 0.0
            n_batches = 0

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                result = model.forward_sparse(batch)

                gain = model.geometry.gain(batch.panel_id)[:, None, None]
                loss_result = loss_fn(
                    result.n_signal,
                    batch.background,
                    batch.is_sparse,
                    batch.sparse_counts,
                    batch.threshold,
                    batch.valid,
                    gain,
                )

                loss_result.loss.backward()
                optimizer.step()

                total_loss += loss_result.loss.item()
                total_t1 += loss_result.term1.item()
                total_t2 += loss_result.term2.item()
                n_batches += 1

            if verbose:
                avg = total_loss / max(n_batches, 1)
                avg1 = total_t1 / max(n_batches, 1)
                avg2 = total_t2 / max(n_batches, 1)
                print(
                    f"    Epoch {epoch+1}/{stage.n_epochs}: "
                    f"loss={avg:.2f} (T1={avg1:.2f} T2={avg2:.2f})"
                )

    return model
