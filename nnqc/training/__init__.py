"""Training modules for nnQC"""

from .train_diffusion import main as train_diffusion

# Import train_autoencoder when available
try:
    from .train_autoencoder import main as train_autoencoder
    __all__ = ["train_diffusion", "train_autoencoder"]
except ImportError:
    __all__ = ["train_diffusion"] 