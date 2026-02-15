"""P2PFL compressions."""

from .dp_strategy import DifferentialPrivacyCompressor, LocalDPCompressor
from .lra_strategy import LowRankApproximation
from .lzma_strategy import LZMACompressor
from .quantization_strategy import PTQuantization
from .topk_strategy import TopKSparsification
from .zlib_strategy import ZlibCompressor

# Import optional strategies with error handling
try:
    from .renyi_dp_strategy import RenyiDifferentialPrivacyCompressor
    RENYI_DP_AVAILABLE = True
except ImportError:
    RenyiDifferentialPrivacyCompressor = None
    RENYI_DP_AVAILABLE = False

try:
    from .sketch_strategy import SketchCompressor, CountSketchCompressor, JLProjectionCompressor
    SKETCH_AVAILABLE = True
except ImportError:
    SketchCompressor = None
    CountSketchCompressor = None
    JLProjectionCompressor = None
    SKETCH_AVAILABLE = False

try:
    from .he_strategy import HomomorphicEncryptionCompressor
    HE_AVAILABLE = True
except ImportError:
    HomomorphicEncryptionCompressor = None
    HE_AVAILABLE = False

try:
    from .he_dp_sketch_compressor import HEDPSketchCompressor
    HE_DP_SKETCH_AVAILABLE = True
except ImportError:
    HEDPSketchCompressor = None
    HE_DP_SKETCH_AVAILABLE = False

# All strategies need to be registered for the manager.
COMPRESSION_STRATEGIES_REGISTRY = {
    "ptq": PTQuantization,
    "topk": TopKSparsification,
    "low_rank": LowRankApproximation,
    "zlib": ZlibCompressor,
    "lzma": LZMACompressor,
    "dp": DifferentialPrivacyCompressor,
    "local_dp": LocalDPCompressor,
}

# Add optional strategies if available
if RENYI_DP_AVAILABLE:
    COMPRESSION_STRATEGIES_REGISTRY["renyi_dp"] = RenyiDifferentialPrivacyCompressor

if SKETCH_AVAILABLE:
    COMPRESSION_STRATEGIES_REGISTRY["countsketch"] = CountSketchCompressor
    COMPRESSION_STRATEGIES_REGISTRY["jl_projection"] = JLProjectionCompressor

if HE_AVAILABLE:
    COMPRESSION_STRATEGIES_REGISTRY["he"] = HomomorphicEncryptionCompressor

if HE_DP_SKETCH_AVAILABLE:
    COMPRESSION_STRATEGIES_REGISTRY["he_dp_sketch"] = HEDPSketchCompressor
