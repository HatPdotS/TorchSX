"""Forward model submodule: pixel-level differentiable forward model for SFX."""

from torchsx.deprecated.forward.coords import RefinableGeometry
from torchsx.deprecated.forward.relp import RelpProfile
from torchsx.deprecated.forward.spectrum import (
    GaussianSpectrum,
    PCASpectrum,
    precompute_spectral_overlap,
)
from torchsx.deprecated.forward.corrections import (
    polarization_correction,
    solid_angle_correction,
    estimate_background,
)
from torchsx.deprecated.forward.forward_model import PixelForwardModel, ForwardResult
from torchsx.deprecated.forward.likelihood import PixelLikelihood
from torchsx.deprecated.forward.batch import ROIBatch, ROIBatchLoader
from torchsx.deprecated.forward.frame_batch import FrameBatchLoader
from torchsx.deprecated.forward.predict import (
    PredictedReflections,
    PredictionDiagnostics,
    predict_reflections,
    compare_predictions,
    update_observations,
)
from torchsx.deprecated.forward.refine import (
    RefinementStage,
    refine,
    refine_sparse,
    DEFAULT_STAGES,
)
from torchsx.deprecated.forward._censored_poisson import (
    CensoredPoissonLoss,
    CensoredPoissonResult,
    compute_censoring_threshold,
)
from torchsx.deprecated.forward._profile_fitted import (
    ProfileFittedLoss,
    ProfileFittedResult,
)
from torchsx.deprecated.forward._soft_assignment import (
    SoftAssignmentLoss,
    SoftAssignmentResult,
)
from torchsx.deprecated.forward._attraction_loss import (
    AttractionLoss,
    AttractionResult,
)
from torchsx.deprecated.forward._sparse_data import (
    SparseFrameStore,
    PixelIndexMapper,
    load_sparse,
)
from torchsx.deprecated.forward.sparse_batch import SparseBatch, SparseBatchLoader, build_frame_to_obs
from torchsx.deprecated.forward._pixel_matching import (
    PixelReflectionPairs,
    SparseForwardResult,
    match_pixels_to_reflections,
)
from torchsx.deprecated.forward._sparse_pixel_loss import SparsePixelLoss, SparsePixelResult
from torchsx.deprecated.forward.sparse_pixel_batch import SparsePixelBatchLoader
