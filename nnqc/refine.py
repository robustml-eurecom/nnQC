"""Accuracy improvements for nnQC reconstruction.

The base model does one DDIM pass per slice from a single noise draw and takes
the argmax. Three cheap, orthogonal upgrades make the reconstructed mask
markedly closer to the true anatomy - which is what a QC score ultimately
depends on:

**Ensembling** (``num_samples``)
    The reverse process is stochastic: different noise seeds give slightly
    different masks. Averaging the decoded *probabilities* over N draws is a
    Monte-Carlo estimate of the posterior mean and cuts boundary variance
    roughly as 1/sqrt(N).

**Post-processing** (``postprocess``)
    Because the model is 2-D and applied slice-by-slice, its errors are
    *z-incoherent*: isolated blobs on one slice, holes that appear on one slice
    and not its neighbours. The obvious remedy - keep only the largest 3-D
    connected component - turned out to be actively harmful on tapered organs
    (apex -0.10, base -0.16 Dice on prostate, base erased outright in 8/25
    cases), so the default clean-up is **hole-filling only**, which is monotone:
    it can add voxels inside a structure but never delete one. Component
    filtering and z-smoothing remain available and off; see :data:`PRESETS`.

Everything here is inference-time only: no retraining, no extra weights.

None of it is assumed to help. ``scripts/benchmark.py`` reports Dice split into
apex/mid/base thirds precisely because a volume-level number hid the fact that
the original clean-up was damaging the extremes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _as_condition(labels, num_classes, onehot=False):
    """Encode a label map [N, 1, H, W] the way training encodes the corrupted mask.

    Legacy multi-class training feeds ``argmax(one_hot)/num_classes`` as a
    single channel; binary training feeds the binary mask. With one-hot
    conditioning (``in_channels == latent_channels + num_classes``) the full
    per-class one-hot stack is fed instead and area-resized downstream, so
    sub-cell class composition survives the latent downsample. Reproduced
    exactly here so refinement passes stay in-distribution.
    """
    if onehot:
        return F.one_hot(labels[:, 0].long().clamp(0, num_classes - 1),
                         num_classes).permute(0, 3, 1, 2).float()
    if num_classes > 1:
        return labels.float() / num_classes
    return (labels > 0.5).float()


def _decode_to_prob(decoded, num_classes):
    """Decoder logits -> per-class probabilities."""
    if num_classes > 1:
        return torch.softmax(decoded.float(), dim=1)
    return torch.sigmoid(decoded.float())


def _prob_to_labels(prob, num_classes):
    if num_classes > 1:
        return prob.argmax(1, keepdim=True).float()
    return (prob > 0.5).float()


@torch.no_grad()
def sample_probabilities(
    *,
    autoencoder,
    unet,
    xa,
    embed,
    scheduler,
    scale_factor,
    scans,
    labels,
    ratios,
    num_classes,
    latent_shape,
    num_steps: int = 5,
    num_samples: int = 1,
    seed: int = 42,
    device=None,
    autocast_enabled: bool = True,
):
    """Reconstruct a batch of slices, returning averaged class probabilities.

    Parameters
    ----------
    scans, labels, ratios
        ``[N, 1, H, W]`` scan slices, ``[N, 1, H, W]`` candidate label maps and
        ``[N, 1]`` slice ratios, all on ``device``.
    latent_shape
        Shape of the latent for this batch, ``[N, C, h, w]``.
    num_samples
        Noise draws to average over (ensembling). 1 reproduces the single-shot
        behaviour of ``nnqc.infer.check``.

    Returns ``[N, C, H, W]`` probabilities (C = num_classes, or 1 if binary).
    """
    device = device or scans.device
    from torch.amp import autocast

    # The CLIP context depends only on the scan and the slice ratio, never on the
    # candidate mask, so it is computed once and reused across every ensemble
    # draw. Recomputing it per draw would re-run the UniMedCLIP vision tower,
    # the most expensive part of a step.
    with autocast("cuda", enabled=autocast_enabled):
        slice_emb = embed(ratios).float()
        context = xa.build_context(scans, slice_emb, mask=labels).float()
        # One-hot conditioning is detected structurally from the UNet: its
        # in_channels then equals latent_channels + num_classes (one channel
        # per class instead of the single legacy ordinal channel).
        onehot = (num_classes > 1
                  and getattr(unet, "in_channels", 0) - latent_shape[1] == num_classes)
        cond = _as_condition(labels, num_classes, onehot=onehot)
        mask_resized = F.interpolate(cond, size=tuple(latent_shape[2:]),
                                     mode="area" if onehot else "nearest")

        acc = None
        for s in range(max(1, num_samples)):
            gen = torch.Generator(device=device).manual_seed(seed + s)
            z_t = torch.randn(tuple(latent_shape), generator=gen, device=device)
            scheduler.set_timesteps(num_steps)
            for t in scheduler.timesteps:
                t_b = torch.full((z_t.shape[0],), int(t.item()), device=device, dtype=torch.long)
                eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=context)
                z_t = scheduler.step(eps, t, z_t)[0]
            decoded = autoencoder.decode_stage_2_outputs(z_t / scale_factor)
            p = _decode_to_prob(decoded, num_classes)
            acc = p if acc is None else acc + p

    return acc / max(1, num_samples)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
@dataclass
class PostprocessConfig:
    """Volume-level clean-up applied to a stack of 2-D predictions.

    Attributes
    ----------
    z_sigma
        Std-dev (in slices) of the Gaussian applied to the probability volume
        along the slice axis. Counteracts the z-incoherence of slice-wise 2-D
        inference. 0 disables. ~0.7 smooths one neighbour either side.
    largest_component
        Keep only the largest 3-D connected component of each foreground class.
        Removes the isolated off-organ blobs that slice-wise models produce.
    min_component_frac
        Alternative//companion to ``largest_component``: drop components smaller
        than this fraction of the largest one. 0 disables.
    fill_holes
        Fill enclosed holes in each foreground class, per slice.
    """

    z_sigma: float = 0.0
    largest_component: bool = False
    min_component_frac: float = 0.0
    fill_holes: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.z_sigma or self.largest_component
                    or self.min_component_frac or self.fill_holes)


def smooth_along_z(prob_volume: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``[C, H, W, D]`` probabilities along the slice axis only.

    Smoothing is linear and mass-preserving per channel, but the argmax that
    follows is not, so the risk is class-dependent: the thinner a class is along
    z relative to its neighbours, the more of its probability can be outvoted by
    them. On the 3-slice-thick nested class in ``tests/test_refine.py`` sigma=0.7
    turns out to be harmless, but that says nothing about anatomy thinner than
    the kernel.

    It stays out of the shipped presets (``z_sigma=0``) because its benefit is
    unproven per task. Note that the alternative once recommended here -
    ``keep_largest_component`` - turned out to be worse, costing apex/base Dice
    on prostate, so neither should be enabled on faith. The ``z_smooth`` preset
    isolates this one variable (it differs from ``ensemble`` by ``z_sigma``
    alone) so a sweep can attribute any difference to it; enable it only if that
    sweep shows a gain on your task.
    """
    if sigma <= 0:
        return prob_volume
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(prob_volume, sigma=sigma, axis=-1, mode="nearest")


def keep_largest_component(labels: np.ndarray, num_classes: int,
                           min_frac: float = 0.0, largest_only: bool = True) -> np.ndarray:
    """Connected-component clean-up on a ``[H, W, D]`` integer label volume."""
    from scipy.ndimage import label as cc_label

    out = labels.copy()
    classes = range(1, num_classes) if num_classes > 1 else [1]
    for c in classes:
        mask = labels == c
        if not mask.any():
            continue
        comp, n = cc_label(mask)
        if n <= 1:
            continue
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        biggest = sizes.argmax()
        if largest_only:
            drop = (comp > 0) & (comp != biggest)
        elif min_frac > 0:
            small = {i for i in range(1, n + 1) if sizes[i] < min_frac * sizes[biggest]}
            drop = np.isin(comp, list(small)) if small else np.zeros_like(mask)
        else:
            continue
        out[drop] = 0
    return out


def fill_holes_per_slice(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Fill enclosed *background* holes inside each foreground class, slice by slice.

    Only voxels that are background in the input are ever filled. That guard
    matters for nested anatomy: the ACDC myocardium is a ring with the LV blood
    pool inside it, so filling class "myocardium" without it would classify the
    entire blood pool as myocardium and erase a whole label. The same applies to
    a tumour inside a liver.
    """
    from scipy.ndimage import binary_fill_holes

    out = labels.copy()
    classes = range(1, num_classes) if num_classes > 1 else [1]
    background = labels == 0
    for c in classes:
        for k in range(labels.shape[-1]):
            sl = labels[..., k] == c
            if not sl.any():
                continue
            filled = binary_fill_holes(sl)
            # never overwrite another foreground class - only true background
            newly = filled & ~sl & background[..., k]
            if newly.any():
                out[..., k][newly] = c
    return out


def postprocess_volume(prob_volume: np.ndarray, num_classes: int,
                       cfg: PostprocessConfig) -> np.ndarray:
    """Apply the configured clean-up.

    ``prob_volume`` is ``[C, H, W, D]`` probabilities. Returns a ``[H, W, D]``
    integer label volume.
    """
    if cfg.z_sigma:
        prob_volume = smooth_along_z(prob_volume, cfg.z_sigma)

    if num_classes > 1:
        labels = prob_volume.argmax(0).astype(np.int16)
    else:
        labels = (prob_volume[0] > 0.5).astype(np.int16)

    if cfg.fill_holes:
        labels = fill_holes_per_slice(labels, num_classes)
    if cfg.largest_component or cfg.min_component_frac:
        labels = keep_largest_component(
            labels, num_classes,
            min_frac=cfg.min_component_frac,
            largest_only=cfg.largest_component)
    return labels


# Presets that trade compute for accuracy. Used by the benchmark and the
# tutorial notebook so the comparison is reproducible.
#
# Note what is NOT in the default clean-up: z-smoothing. It looked like the
# obvious fix for slice-wise incoherence, but it silently destroys thin classes
# (see smooth_along_z). Connected-component removal buys the same z-coherence
# with no such failure mode, so that is what the presets use; `z_smooth` stays
# available as an explicitly opt-in arm for the per-task sweep.
# Connected-component filtering is OFF by default, against the intuition that
# started this module. Measured on prostate (results/prostate_mid_fixedbench.json,
# 25 paired volume/severity pairs), `largest_component=True` cost
#   apex  -0.1023 Dice (worse in 19/25)
#   base  -0.1559 Dice (worse in 19/25)
#   mid   -0.0285 Dice (better in 16/25)
# and erased the base structure outright in 8/25 cases.
#
# The cause is structural rather than a bad threshold. At apex and base the
# cross-section is genuinely small - in the test fixture the apex island is 16
# voxels against a 432-voxel body, under 4% - so *any* keep-the-big-ones rule
# either spares the specks it exists to remove or amputates real anatomy. Since
# the measured net effect was negative even at mid, the default keeps only
# hole-filling, which is monotone: it can add voxels inside a structure but
# never delete one.
#
# `min_component_frac` and `largest_component` remain available for tasks with a
# single compact organ and no taper (spleen, kidney), where they are safe - but
# they should be measured per task, not assumed.
_CLEANUP = PostprocessConfig(fill_holes=True)

PRESETS = {
    # Paper-faithful baseline: one draw, raw argmax.
    "baseline": {"num_steps": 5, "num_samples": 1,
                 "postprocess": PostprocessConfig()},
    # Post-processing only - no extra network passes at all.
    "cleanup": {"num_steps": 5, "num_samples": 1,
                "postprocess": _CLEANUP},
    # Average 5 noise draws, then clean up. ~5x the sampling cost.
    "ensemble": {"num_steps": 5, "num_samples": 5,
                 "postprocess": _CLEANUP},
    # Longer DDIM chain, to test the README's "5 steps is best" claim.
    "long_chain": {"num_steps": 25, "num_samples": 3,
                   "postprocess": _CLEANUP},
    # Steps-only ladder. `long_chain` varies steps, samples AND postprocessing at
    # once, so it cannot answer "does cardiac need more steps?" - these can: they
    # differ from `baseline` by num_steps alone, so any difference is attributable
    # to the length of the DDIM chain.
    #
    # Motivation: cardiac's reconstruction is per-pixel class speckle (see
    # results/inspect/cardiac_*.png), and speckle is what an under-converged
    # denoiser leaves behind. 5 steps is the tuned default for the binary case;
    # it was never separately validated on the 4-class latent.
    # The first sweep only tested 5 and UP and found monotonic decline, which
    # leaves the optimum unbracketed from below - 5 might itself be past it.
    "steps1": {"num_steps": 1, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps2": {"num_steps": 2, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps3": {"num_steps": 3, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps4": {"num_steps": 4, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps5": {"num_steps": 5, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps10": {"num_steps": 10, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps25": {"num_steps": 25, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps50": {"num_steps": 50, "num_samples": 1, "postprocess": PostprocessConfig()},
    "steps100": {"num_steps": 100, "num_samples": 1, "postprocess": PostprocessConfig()},
    # Opt-in arm isolating z-smoothing. It deliberately does NOT also enable
    # component filtering: bundling two interventions makes a bad result
    # uninterpretable, and component filtering is already known to cost apex/base
    # Dice here. This arm differs from `ensemble` by z_sigma alone, so any
    # difference between them is attributable.
    "z_smooth": {"num_steps": 5, "num_samples": 5,
                 "postprocess": PostprocessConfig(z_sigma=0.5, fill_holes=True)},
}
