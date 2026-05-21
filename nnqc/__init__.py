"""nnQC - segmentation quality control with a 2D latent diffusion model.

A trained nnQC model reconstructs a *correct* segmentation mask from a
*corrupted* one, conditioned on the underlying CT/MR scan. The Dice between the
input mask and the reconstruction is the QC signal.

Quick start (notebook)::

    import nnqc
    nnqc.train_autoencoder(task="prostate", epochs=500, lr=5e-5)
    nnqc.train_diffusion(task="prostate", epochs=4000, lr=2.5e-5,
                         scheduler="cosine", warmup_dice_epochs=100)
    nnqc.evaluate(task="prostate", num_steps=5)

Quick start (CLI)::

    nnqc train-autoencoder --task prostate --epochs 500
    nnqc train-diffusion   --task prostate --epochs 4000 --scheduler cosine
    nnqc evaluate          --task prostate --num-steps 5
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Lightweight, torch-free helpers are safe to import eagerly.
from nnqc.config import available_tasks, resolve_config

# Heavy entry points (pull in torch/monai) are loaded lazily on first access.
_LAZY = {
    "train_autoencoder": ("nnqc.train", "train_autoencoder"),
    "train_diffusion": ("nnqc.train", "train_diffusion"),
    "evaluate": ("nnqc.evaluate", "evaluate"),
    "check": ("nnqc.infer", "check"),
    "QCResult": ("nnqc.infer", "QCResult"),
}

__all__ = ["__version__", "available_tasks", "resolve_config", *_LAZY]

if TYPE_CHECKING:  # for type checkers / IDEs only
    from nnqc.evaluate import evaluate
    from nnqc.infer import QCResult, check
    from nnqc.train import train_autoencoder, train_diffusion


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module 'nnqc' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
