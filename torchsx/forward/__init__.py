"""Forward model submodule: pixel-level differentiable forward model for SFX."""

from torchsx.forward.coords import RefinableGeometry
from torchsx.forward.relp import RelpProfile
from torchsx.forward.spectrum import (
    GaussianSpectrum,
    PCASpectrum,
    precompute_spectral_overlap,
)
from torchsx.forward.corrections import (
    polarization_correction,
    solid_angle_correction,
    estimate_background,
)
from torchsx.forward.forward_model import PixelForwardModel, ForwardResult
from torchsx.forward.likelihood import PixelLikelihood
from torchsx.forward.batch import ROIBatch, ROIBatchLoader
from torchsx.forward.frame_batch import FrameBatchLoader
from torchsx.forward.predict import (
    PredictedReflections,
    PredictionDiagnostics,
    predict_reflections,
    compare_predictions,
    update_observations,
)
from torchsx.forward.refine import (
    RefinementStage,
    refine,
    refine_sparse,
    DEFAULT_STAGES,
)
from torchsx.forward._censored_poisson import (
    CensoredPoissonLoss,
    CensoredPoissonResult,
    compute_censoring_threshold,
)
from torchsx.forward._profile_fitted import (
    ProfileFittedLoss,
    ProfileFittedResult,
)
from torchsx.forward._soft_assignment import (
    SoftAssignmentLoss,
    SoftAssignmentResult,
)
from torchsx.forward._attraction_loss import (
    AttractionLoss,
    AttractionResult,
)
from torchsx.forward._sparse_data import (
    SparseFrameStore,
    PixelIndexMapper,
    load_sparse,
)
from torchsx.forward.sparse_batch import SparseBatch, SparseBatchLoader, build_frame_to_obs
from torchsx.forward._pixel_matching import (
    PixelReflectionPairs,
    SparseForwardResult,
    match_pixels_to_reflections,
)
from torchsx.forward._sparse_pixel_loss import SparsePixelLoss, SparsePixelResult
from torchsx.forward.sparse_pixel_batch import SparsePixelBatchLoader
