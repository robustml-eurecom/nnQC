"""nnQC - segmentation quality-control with a 2D latent diffusion model.

A trained nnQC model reconstructs a *correct* segmentation mask from a
*corrupted* one, conditioned on the underlying CT/MR scan. The Dice between
the input mask and the reconstruction is the QC signal.

Top-level helpers exposed here:

    from nnqc.corruptions import corrupt_ohe_masks_v2
    from nnqc.utils import (
        compute_spacing, define_instance, KL_loss, setup_ddp,
        prepare_general_dataloader, prepare_msd_dataloader,
    )
    from nnqc.xa import CLIPCrossAttentionGrid
"""

__version__ = "0.1.0"
