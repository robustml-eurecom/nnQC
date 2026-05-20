"""
Anatomically realistic mask corruptions for nnQC training (corruptions_v2).

Ported from PanQC/src/panqc/data/corruptions_v2.py and adapted from 3-D
single-organ binary masks to the 2-D multi-class one-hot masks used by the
main nnQC diffusion model.

Core idea: operate on the Signed Distance Transform (SDT) of the mask, add
smooth Gaussian-filtered perturbations, and re-threshold. This guarantees
smooth, organic boundaries that mimic real segmentation-model failures -
NOT the rectangular / axis-aligned holes produced by the old
`corrupt_ohe_masks`.

Public API:
    corrupt_ohe_masks_v2(ohe_masks, corruption_prob=1.0, config=None)
        Drop-in replacement for `utils.corrupt_ohe_masks`. Takes a one-hot
        tensor [B, C, H, W] and returns a corrupted one-hot tensor of the
        same shape, device and dtype.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as ndimage_label,
)


# ---------------------------------------------------------------------------
# Primitives - operate on N-D numpy binary arrays (here used in 2-D)
# ---------------------------------------------------------------------------

def signed_distance_transform(mask):
    """Signed distance transform: positive inside, negative outside."""
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return -distance_transform_edt(~mask_bool)
    if mask_bool.all():
        return distance_transform_edt(mask_bool)
    dt_in = distance_transform_edt(mask_bool)
    dt_out = distance_transform_edt(~mask_bool)
    return dt_in - dt_out


def elastic_boundary_deformation(mask, sigma=8.0, magnitude=4.0):
    """Deform the boundary with a smooth random displacement field.

    Perturbs the signed distance field with Gaussian-smoothed noise and
    re-thresholds at zero - organic over/under-segmentation.
    """
    if not mask.any():
        return mask.copy()
    sdt = signed_distance_transform(mask)
    noise = gaussian_filter(np.random.randn(*mask.shape) * magnitude, sigma=sigma)
    return (sdt + noise > 0).astype(mask.dtype)


def smooth_boundary_perturbation(mask, noise_std=2.0, smooth_sigma=10.0):
    """Gently perturb the boundary with low-frequency noise.

    Lighter than `elastic_boundary_deformation` - minor boundary imprecision
    typical of a well-performing segmentor.
    """
    if not mask.any():
        return mask.copy()
    sdt = signed_distance_transform(mask)
    noise = gaussian_filter(np.random.randn(*mask.shape) * noise_std, sigma=smooth_sigma)
    return (sdt + noise > 0).astype(mask.dtype)


def smooth_erosion(mask, fraction=0.2):
    """Smoothly under-segment by thresholding the distance transform.

    Unlike `binary_erosion`, keeps smooth boundaries and erodes proportional
    to organ size.
    """
    if not mask.any():
        return mask.copy()
    dt = distance_transform_edt(mask.astype(bool))
    max_dt = dt.max()
    if max_dt == 0:
        return mask.copy()
    return (dt > max_dt * fraction).astype(mask.dtype)


def smooth_dilation(mask, voxels=3.0, smooth_sigma=2.0):
    """Smoothly over-segment by expanding the boundary, with smooth noise."""
    if not mask.any():
        return mask.copy()
    bg_dt = distance_transform_edt(~mask.astype(bool))
    expanded = bg_dt <= voxels
    if smooth_sigma > 0:
        noise = gaussian_filter(np.random.randn(*mask.shape), sigma=smooth_sigma)
        boundary = (bg_dt > 0) & (bg_dt < voxels * 2)
        sdt_expanded = (distance_transform_edt(expanded)
 - distance_transform_edt(~expanded))
        sdt_expanded[boundary] += noise[boundary] * 1.5
        return (sdt_expanded > 0).astype(mask.dtype)
    return expanded.astype(mask.dtype)


def blob_false_positive(mask, num_blobs=1, size_fraction=0.25):
    """Add smooth, blob-shaped false positives near the organ.

    Generates rounded, anatomically plausible extra regions with
    Gaussian-filtered noise - not rectangular patches. N-D generic.
    """
    result = mask.copy()
    if not mask.any():
        return result

    ndim = mask.ndim
    coords = np.where(mask)
    bbox_min = np.array([c.min() for c in coords])
    bbox_max = np.array([c.max() for c in coords])
    organ_extent = bbox_max - bbox_min + 1
    bg_dt = distance_transform_edt(~mask.astype(bool))

    for _ in range(num_blobs):
        # Place blob at moderate distance from the organ.
        min_dist = 3
        max_dist = max(5, int(organ_extent.mean() * 0.4))
        suitable = (bg_dt > min_dist) & (bg_dt < max_dist)

        # Restrict to the general organ neighbourhood (not far away).
        for d in range(ndim):
            margin = int(organ_extent[d] * 0.6)
            lo = max(0, bbox_min[d] - margin)
            hi = min(mask.shape[d], bbox_max[d] + margin)
            coord_mask = np.zeros(mask.shape, dtype=bool)
            slc = [slice(None)] * ndim
            slc[d] = slice(lo, hi)
            coord_mask[tuple(slc)] = True
            suitable &= coord_mask

        if not suitable.any():
            continue

        candidates = np.where(suitable)
        idx = np.random.randint(len(candidates[0]))
        center = np.array([candidates[d][idx] for d in range(ndim)])

        blob_radius = np.array([
            max(2, int(s * size_fraction * random.uniform(0.4, 1.2)))
            for s in organ_extent
        ])

        patch_size = blob_radius * 3
        noise = gaussian_filter(np.random.randn(*patch_size), sigma=blob_radius * 0.4)
        blob = noise > np.percentile(noise, 65)

        # Keep only the largest connected component of the blob.
        labeled, n = ndimage_label(blob)
        if n > 0:
            sizes = [np.sum(labeled == i) for i in range(1, n + 1)]
            blob = labeled == (int(np.argmax(sizes)) + 1)

        # Paste the blob into the volume, clipping at the borders.
        sl_dst, sl_src = [], []
        for d in range(ndim):
            vol_start = int(center[d] - patch_size[d] // 2)
            vol_end = vol_start + int(patch_size[d])
            dst_start = max(0, vol_start)
            dst_end = min(mask.shape[d], vol_end)
            src_start = dst_start - vol_start
            sl_dst.append(slice(dst_start, dst_end))
            sl_src.append(slice(src_start, src_start + (dst_end - dst_start)))

        result[tuple(sl_dst)] = np.maximum(
            result[tuple(sl_dst)],
            blob[tuple(sl_src)].astype(result.dtype),
        )

    return result


def smooth_split(mask, smooth_sigma=6.0):
    """Split the mask along a smooth random surface.

    Simulates topology errors where one region is broken into pieces, or one
    piece is dropped entirely. Uses a random Gaussian field as the cut.
    """
    if not mask.any():
        return mask.copy()
    field = gaussian_filter(np.random.randn(*mask.shape), sigma=smooth_sigma)
    mask_vals = field[mask.astype(bool)]
    if len(mask_vals) == 0:
        return mask.copy()
    threshold = np.median(mask_vals)
    keep = field > threshold if random.random() < 0.5 else field < threshold
    result = mask.copy()
    result[~keep] = 0
    return result


# ---------------------------------------------------------------------------
# Composite single-class 2-D corruption
# ---------------------------------------------------------------------------

# Defaults tuned for ~256x256 slices. Ranges are sampled uniformly.
DEFAULT_CFG = {
    "elastic_prob": 0.5,        "elastic_sigma": (5.0, 12.0),  "elastic_magnitude": (2.0, 6.0),
    "boundary_prob": 0.4,       "boundary_noise_std": (1.0, 3.0), "boundary_smooth_sigma": (6.0, 14.0),
    "erosion_prob": 0.3,        "erosion_fraction": (0.05, 0.30),
    "dilation_prob": 0.3,       "dilation_voxels": (1.5, 5.0),
    "blob_fp_prob": 0.35,       "blob_num": (1, 2),            "blob_size": (0.10, 0.35),
    "split_prob": 0.15,
    "max_operations": 3,
}


def corrupt_binary_2d(mask, config=None):
    """Apply 1-3 realistic corruptions to a single 2-D binary mask.

    Args:
        mask: 2-D numpy array (binary, float32).
        config: optional dict overriding `DEFAULT_CFG`.
    Returns:
        Corrupted 2-D binary float32 array.
    """
    cfg = dict(DEFAULT_CFG)
    if config:
        cfg.update(config)

    if not mask.any():
        return mask.copy()

    ops = []
    if random.random() < cfg["elastic_prob"]:
        s = random.uniform(*cfg["elastic_sigma"])
        m = random.uniform(*cfg["elastic_magnitude"])
        ops.append(lambda x, s=s, m=m: elastic_boundary_deformation(x, sigma=s, magnitude=m))
    if random.random() < cfg["boundary_prob"]:
        n = random.uniform(*cfg["boundary_noise_std"])
        s = random.uniform(*cfg["boundary_smooth_sigma"])
        ops.append(lambda x, n=n, s=s: smooth_boundary_perturbation(x, noise_std=n, smooth_sigma=s))
    if random.random() < cfg["erosion_prob"]:
        f = random.uniform(*cfg["erosion_fraction"])
        ops.append(lambda x, f=f: smooth_erosion(x, fraction=f))
    if random.random() < cfg["dilation_prob"]:
        v = random.uniform(*cfg["dilation_voxels"])
        ops.append(lambda x, v=v: smooth_dilation(x, voxels=v))
    if random.random() < cfg["blob_fp_prob"]:
        nb = random.randint(*cfg["blob_num"])
        sf = random.uniform(*cfg["blob_size"])
        ops.append(lambda x, nb=nb, sf=sf: blob_false_positive(x, num_blobs=nb, size_fraction=sf))
    if random.random() < cfg["split_prob"]:
        ops.append(lambda x: smooth_split(x))

    # Always apply at least one operation.
    if not ops:
        n = random.uniform(*cfg["boundary_noise_std"])
        s = random.uniform(*cfg["boundary_smooth_sigma"])
        ops.append(lambda x, n=n, s=s: smooth_boundary_perturbation(x, noise_std=n, smooth_sigma=s))

    if len(ops) > cfg["max_operations"]:
        ops = random.sample(ops, cfg["max_operations"])

    corrupted = mask.astype(np.float32).copy()
    for fn in ops:
        corrupted = fn(corrupted).astype(np.float32)
    return (corrupted > 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Public wrapper - multi-class one-hot, drop-in for utils.corrupt_ohe_masks
# ---------------------------------------------------------------------------

def corrupt_ohe_masks_v2(ohe_masks, corruption_prob=1.0, config=None):
    """Apply anatomically realistic corruptions to one-hot 2-D masks.

    Operates on the recovered integer label map (argmax), corrupts each
    foreground class independently, then recombines into a single label map
    so the result is *always* a valid one-hot encoding (no overlapping
    classes, no spurious background ties - unlike the channel-wise
    `corrupt_ohe_masks`).

    Args:
        ohe_masks: torch.Tensor [B, C, H, W], one-hot (channel 0 = background).
        corruption_prob: per-sample probability of applying any corruption.
        config: optional dict overriding `DEFAULT_CFG`.
    Returns:
        torch.Tensor [B, C, H, W], corrupted one-hot, same device/dtype.
    """
    device, dtype = ohe_masks.device, ohe_masks.dtype
    B, C, H, W = ohe_masks.shape

    # Binary single-channel mask (num_classes=1): no OHE / argmax path.
    if C == 1:
        np_in = (ohe_masks[:, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
        np_out = np.zeros_like(np_in)
        for b in range(B):
            if random.random() > corruption_prob or not np_in[b].any():
                np_out[b] = np_in[b]
                continue
            np_out[b] = corrupt_binary_2d(np_in[b], config)
        out = torch.from_numpy(np_out).to(device).unsqueeze(1).to(dtype)
        return out

    # Multi-class one-hot path: operate on the recovered label map.
    labels = ohe_masks.argmax(dim=1).cpu().numpy()  # [B, H, W]
    out = np.zeros_like(labels)

    for b in range(B):
        if random.random() > corruption_prob:
            out[b] = labels[b]
            continue
        new_label = np.zeros((H, W), dtype=labels.dtype)
        for c in range(1, C):  # foreground classes only
            binary = (labels[b] == c).astype(np.float32)
            if binary.any():
                binary = corrupt_binary_2d(binary, config)
            new_label[binary > 0.5] = c  # later classes win on overlap
        out[b] = new_label

    out_t = torch.from_numpy(out).to(device).long()
    ohe = F.one_hot(out_t, num_classes=C).permute(0, 3, 1, 2).contiguous()
    return ohe.to(dtype)
