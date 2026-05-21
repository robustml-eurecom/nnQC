"""One-call inference / quality control for a trained nnQC model.

    import nnqc
    result = nnqc.check("scan.nii.gz", "candidate_mask.nii.gz", task="prostate")
    print(result.qc_score)            # volume Dice(input mask, reconstruction)
    result.save("reconstruction.nii.gz")

``check`` takes a scan and a *candidate* segmentation, preprocesses both exactly
like training (orientation, slice/foreground crop, resize, intensity scaling),
asks the diffusion model to reconstruct the mask it believes is correct, and
returns the agreement between the candidate and the reconstruction as the QC
signal. The reconstruction is mapped back onto the input volume's grid so it can
be saved or compared directly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SpatialCropd,
)
from monai.utils import set_determinism

from nnqc.config import resolve_config
from nnqc.utils import define_instance
from nnqc.xa import CLIPCrossAttentionGrid


@dataclass
class QCResult:
    """Result of :func:`check`.

    Attributes
    ----------
    qc_score
        Volume-level Dice between the input mask and the reconstruction
        (foreground union). 1.0 = perfect agreement, low = likely a bad mask.
    qc_score_per_class
        Per-class Dice (multi-class models only).
    slice_scores, slice_ratios
        Per-slice Dice and normalized slice position (apex=0 ... base=1).
    reconstruction
        Reconstructed label map as a MONAI ``MetaTensor`` on the *input* grid
        (same shape/affine as the candidate mask), or ``None`` if the inverse
        mapping failed (then ``reconstruction_model_space`` is still set).
    reconstruction_model_space
        Reconstruction in the preprocessed model space ``[H, W, n_slices]``.
    """

    qc_score: float
    qc_score_per_class: dict[int, float] = field(default_factory=dict)
    slice_scores: np.ndarray | None = None
    slice_ratios: np.ndarray | None = None
    reconstruction: MetaTensor | None = None
    reconstruction_model_space: np.ndarray | None = None

    def save(self, path: str | Path) -> str:
        """Write the (original-grid) reconstruction to a NIfTI file."""
        if self.reconstruction is None:
            raise RuntimeError(
                "No original-grid reconstruction available (inverse mapping failed). "
                "Use reconstruction_model_space instead."
            )
        import nibabel as nib

        arr = self.reconstruction.detach().cpu().numpy()
        arr = np.squeeze(arr)  # drop channel dim
        affine = np.asarray(self.reconstruction.affine.detach().cpu().numpy())
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))
        return str(path)


def _intensity_transform(modality):
    if modality == "mri":
        return ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1)
    if modality == "ct":
        return ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1)
    if modality == "us":
        return NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    raise ValueError(f"Unknown modality {modality!r}; expected mri, ct or us.")


def _nonzero_slice_bounds(label):  # label: [1, H, W, D]
    proj = label.abs().sum(dim=(0, 1, 2))
    nz = torch.nonzero(proj > 0).flatten()
    if nz.numel() == 0:
        return 0, int(label.shape[-1])
    return int(nz[0].item()), int(nz[-1].item()) + 1


def _binary_dice(a, b, eps=1e-6):
    a = (a > 0.5).float()
    b = (b > 0.5).float()
    return float(((2 * (a * b).sum() + eps) / (a.sum() + b.sum() + eps)).item())


@torch.no_grad()
def check(
    image,
    mask,
    config=None,
    env=None,
    task=None,
    *,
    checkpoint: str = "best",
    num_steps: int = 5,
    inference_batch: int = 16,
    device=None,
    seed: int = 42,
    return_volume: bool = True,
    **overrides,
) -> QCResult:
    """Run nnQC on one scan + candidate mask pair.

    Parameters
    ----------
    image, mask
        Paths to the scan and candidate-segmentation volumes (NIfTI).
    config, env, task
        Model configuration, as in the other entry points (a bundled ``task=``
        preset, or an explicit ``config=``/``env=`` pair). ``model_dir`` must
        contain the trained checkpoints.
    checkpoint
        ``"best"`` (diffusion_unet.pt) or ``"last"`` (diffusion_unet_last.pt).
    num_steps
        DDIM sampling steps (5 is the recommended default).
    inference_batch
        Slices per forward pass.
    return_volume
        If False, skip mapping the reconstruction back to the input grid
        (faster; ``reconstruction`` will be ``None``).
    """
    cfg = resolve_config(config, env, task, stage="diffusion", overrides=overrides)
    set_determinism(seed)
    dev = torch.device("cuda") if device is None else torch.device(
        device if isinstance(device, str) else f"cuda:{int(device)}")
    num_classes = cfg.num_classes
    patch = cfg.diffusion_train["patch_size"]
    print(f"[nnqc] check | device={dev} num_classes={num_classes} steps={num_steps}")

    # --- preprocessing (invertible geometric chain so we can map back) -------
    if num_classes == 1:
        label_fn = lambda x: torch.where(x[0] > 0.5, 1, 0)  # noqa: E731
    else:
        label_fn = lambda x: x[0]  # noqa: E731
    # Non-geometric prep (load, channel select, binarize, type). These do not
    # change the voxel grid, so they are excluded from the inverse mapping.
    pre = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x[cfg.channel]),
        Lambdad(keys="label", func=label_fn),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"]),
    ])
    data = pre({"image": str(image), "label": str(mask)})

    # Geometric chain - each transform is invertible and affine-aware, so the
    # reconstruction can be mapped back exactly. Kept as references for inverse.
    orient = Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True)
    data = orient(data)

    start, end = _nonzero_slice_bounds(data["label"])
    H, W = data["label"].shape[1], data["label"].shape[2]
    slice_crop = SpatialCropd(keys=["image", "label"], roi_start=[0, 0, start],
                              roi_end=[H, W, end], allow_missing_keys=True)
    crop_fg = CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True)
    resize = Resized(keys=["image", "label"], spatial_size=(patch[0], patch[1], -1),
                     mode=("area", "nearest"), allow_missing_keys=True)
    intensity = _intensity_transform(cfg.modality)

    data = slice_crop(data)
    data = crop_fg(data)
    data = resize(data)
    label_proc = data["label"]  # [1, H', W', D]  (carries applied_operations)
    data = intensity(data)

    scan_vol = data["image"].to(dev).float()      # [1, P, P, D]
    mask_vol = label_proc.to(dev).float()         # [1, P, P, D]  integer label map
    n_slices = scan_vol.shape[-1]
    scans = scan_vol.permute(3, 0, 1, 2).contiguous()   # [D, 1, P, P]
    masks = mask_vol.permute(3, 0, 1, 2).contiguous()    # [D, 1, P, P]
    ratios = (torch.arange(n_slices, device=dev).float() / max(n_slices - 1, 1)).unsqueeze(1)

    # --- models --------------------------------------------------------------
    autoencoder = define_instance(cfg, "autoencoder_def").to(dev).eval()
    autoencoder.load_state_dict(torch.load(
        f"{cfg.model_dir}/autoencoder.pt", map_location=dev, weights_only=True))
    unet = define_instance(cfg, "diffusion_def").to(dev).eval()
    ck = "diffusion_unet.pt" if checkpoint == "best" else "diffusion_unet_last.pt"
    unet.load_state_dict(torch.load(f"{cfg.model_dir}/{ck}", map_location=dev, weights_only=True))
    xa = CLIPCrossAttentionGrid(
        output_dim=cfg.diffusion_def["cross_attention_dim"], grid_reduction="column_softmax").to(dev).eval()
    xa.load_state_dict(torch.load(
        f"{cfg.model_dir}/{'xa.pt' if checkpoint == 'best' else 'xa_last.pt'}",
        map_location=dev, weights_only=True))
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.GELU(),
        torch.nn.Linear(32, cfg.diffusion_def["cross_attention_dim"])).to(dev).eval()
    embed.load_state_dict(torch.load(
        f"{cfg.model_dir}/{'embed.pt' if checkpoint == 'best' else 'embed_last.pt'}",
        map_location=dev, weights_only=True))

    # candidate mask as the model's conditioning channel
    if num_classes > 1:
        cond = masks / num_classes
    else:
        cond = (masks > 0.5).float()

    # scale factor, recomputed from the encoded candidate (as in eval)
    if num_classes > 1:
        ohe = F.one_hot(masks[:, 0].long().clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    else:
        ohe = (masks > 0.5).float()
    z_probe = autoencoder.encode_stage_2_inputs(ohe[:1])
    scale_factor = 1.0 / torch.std(z_probe)

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=cfg.NoiseScheduler["beta_start"],
        beta_end=cfg.NoiseScheduler["beta_end"],
        clip_sample=cfg.NoiseScheduler["clip_sample"],
    )

    # --- DDIM sampling, batched over slices ----------------------------------
    pgt = torch.zeros_like(masks)  # [D, 1, P, P] label map
    for s in range(0, n_slices, inference_batch):
        e = min(s + inference_batch, n_slices)
        scan_b, cond_b, r_b = scans[s:e], cond[s:e], ratios[s:e]
        slice_emb = embed(r_b).float()
        c = xa(scan_b, ext_features=slice_emb)[0].float().unsqueeze(1)
        zsh = autoencoder.encode_stage_2_inputs(ohe[s:e]).shape
        mask_resized = F.interpolate(cond_b, size=zsh[2:], mode="nearest")
        gen = torch.Generator(device=dev).manual_seed(seed + s)
        z_t = torch.randn(zsh, generator=gen, device=dev)
        scheduler.set_timesteps(num_steps)
        for t in scheduler.timesteps:
            t_b = torch.full((z_t.shape[0],), int(t.item()), device=dev, dtype=torch.long)
            eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=c)
            z_t = scheduler.step(eps, t, z_t)[0]
        decoded = autoencoder.decode_stage_2_outputs(z_t / scale_factor)
        if num_classes > 1:
            pgt[s:e, 0] = decoded.argmax(1).float()
        else:
            pgt[s:e, 0] = (decoded[:, 0].sigmoid() > 0.5).float()

    # --- QC scores (input mask vs reconstruction) ----------------------------
    slice_scores = np.zeros(n_slices, dtype=np.float32)
    for i in range(n_slices):
        a = (masks[i, 0] > 0.5) if num_classes == 1 else (masks[i, 0] > 0.5)
        b = (pgt[i, 0] > 0.5) if num_classes == 1 else (pgt[i, 0] > 0.5)
        slice_scores[i] = _binary_dice(a.float(), b.float())
    qc_score = _binary_dice((masks > 0.5).float(), (pgt > 0.5).float())

    per_class = {}
    if num_classes > 1:
        for cidx in range(1, num_classes):
            per_class[cidx] = _binary_dice((masks == cidx).float(), (pgt == cidx).float())

    result = QCResult(
        qc_score=qc_score,
        qc_score_per_class=per_class,
        slice_scores=slice_scores,
        slice_ratios=ratios.squeeze(1).detach().cpu().numpy(),
        reconstruction_model_space=pgt.permute(1, 2, 3, 0)[0].detach().cpu().numpy(),
    )

    # --- map reconstruction back onto the input grid -------------------------
    if return_volume:
        try:
            pgt_vol = pgt.permute(1, 2, 3, 0).contiguous()  # [1, P, P, D]
            pgt_meta = MetaTensor(
                pgt_vol.cpu(),
                meta=copy.deepcopy(label_proc.meta),
                applied_operations=copy.deepcopy(label_proc.applied_operations),
            )
            inv = {"label": pgt_meta}
            for t in (resize, crop_fg, slice_crop, orient):
                inv = t.inverse(inv)
            result.reconstruction = inv["label"]
        except Exception as exc:  # pragma: no cover - geometry edge cases
            print(f"[nnqc] warning: inverse mapping to input grid failed ({exc}); "
                  "returning model-space reconstruction only.")

    print(f"[nnqc] QC score (Dice input vs reconstruction) = {qc_score:.4f}")
    return result
