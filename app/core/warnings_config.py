"""
Configuration for suppressing known warnings and optimizing for CPU usage.
"""

import warnings
import os
import logging

# Suppress known warnings
def configure_warnings():
    """Configure warnings for clean production logs."""

    # Suppress FP16 CPU warnings
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
    warnings.filterwarnings("ignore", message=".*FP32 instead.*")

    # Suppress pyannote version warnings
    warnings.filterwarnings("ignore", message=".*was trained with pyannote.audio.*")
    warnings.filterwarnings("ignore", message=".*yours is.*")

    # Suppress embedding shape warnings (temporary)
    warnings.filterwarnings("ignore", message=".*Shape mismatch in cosine_sim.*")

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Suppress torch warnings
    warnings.filterwarnings("ignore", message=".*torch.distributed.*")
    warnings.filterwarnings("ignore", message=".*CUDA.*")

    print("✅ Warning filters configured for clean logs")


def configure_cpu_optimizations():
    """Configure optimizations for CPU-only inference."""

    # Disable CUDA warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set torch to CPU mode
    os.environ["TORCH_DEVICE"] = "cpu"

    # Optimize for CPU inference
    os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
    os.environ["MKL_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))

    print("✅ CPU optimizations configured")


def configure_model_compatibility():
    """Configure model compatibility settings."""

    # Disable strict version checking
    os.environ["PYANNOTE_STRICT_VERSION"] = "false"

    # Force CPU inference for all models
    os.environ["FORCE_CPU"] = "true"

    print("✅ Model compatibility configured")


def setup_production_environment():
    """Setup clean production environment."""

    # Configure all optimizations
    configure_warnings()
    configure_cpu_optimizations()
    configure_model_compatibility()

    # Set clean logging format
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    print("🚀 Production environment configured")