"""
nnQC: Neural Network Quality Control for Medical Image Segmentation

A comprehensive toolkit for training, evaluating, and improving medical image 
segmentation models using diffusion-based approaches.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import training
from . import inference
from . import evaluation
from . import utils
from . import models

# Main API functions
from .training.train_autoencoder import train_autoencoder
from .training.train_diffusion import train_diffusion
from .inference.inference import evaluate_validation_set
from .evaluation.metrics import compute_metrics_for_validation

__all__ = [
    "train_autoencoder",
    "train_diffusion", 
    "evaluate_validation_set",
    "compute_metrics_for_validation",
    "training",
    "inference", 
    "evaluation",
    "utils",
    "models"
] 