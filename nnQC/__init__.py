"""
nnQC: Neural Network Quality Control for Medical Image Segmentation

A comprehensive toolkit for training, evaluating, and improving medical image 
segmentation models using diffusion-based approaches.
"""

__version__ = "0.1.0"
__author__ = "Vincenzo Marciano"
__email__ = "vincenzo.marciano@eurecom.fr"

from . import training
from .inference import inference
from .inference import evaluation
from .utils import utils, visualize_2d_image, download_msd_data
from .models import xa

# Main API functions
from .training import train_autoencoder
from .training import train_diffusion


__all__ = [
    "train_autoencoder",
    "train_diffusion", 
    "evaluate_validation_set",
    "compute_metrics_for_validation",
    "training",
    "inference", 
    "evaluation",
    "utils",
    "visualize_2d_image",
    "download_msd_data",
    "xa"
] 