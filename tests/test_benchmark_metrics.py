"""Tests for the benchmark's scoring, written after two of them lied.

Both bugs were the same shape - an aggregate number that looked reasonable while
hiding the thing being measured:

* `multiclass_dice` averaged in a 0 for classes absent from the reference, so
  MSD prostate_18 (no transition zone) scored 0.18 instead of ~0.6 and dragged
  the whole benchmark mean down. "Nothing to reconstruct" was being reported as
  "reconstructed wrongly".
* `corrupt_volume` was too mild to produce bad candidates at all - every
  severity landed between 0.81 and 0.99 true Dice - so the QC calibration
  correlation came out ~0 *by construction* and looked like a model failure.

`zone_dice` exists for the same reason: a volume-level Dice is dominated by
mid-organ slices and can look healthy while apex and base have collapsed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.benchmark import (
    corrupt_volume,
    dice,
    hallucinated_classes,
    multiclass_dice,
    zone_dice,
)

H = W = 32
D = 9
NC = 3


def _tapered_volume():
    """Small-large-small along z, with a nested second class - prostate-like."""
    gt = np.zeros((H, W, D), np.int16)
    yy, xx = np.ogrid[:H, :W]
    for k in range(D):
        taper = 1.0 - abs(k - D // 2) / (D / 2)
        r1, r2 = int(3 + 9 * taper), int(1 + 4 * taper)
        gt[..., k][(yy - 16) ** 2 + (xx - 16) ** 2 <= r1 * r1] = 1
        gt[..., k][(yy - 16) ** 2 + (xx - 16) ** 2 <= r2 * r2] = 2
    return gt


# --------------------------------------------------------------------------- #
# multiclass_dice: absent classes
# --------------------------------------------------------------------------- #
def test_absent_class_in_reference_is_not_scored_zero():
    """The prostate_18 bug: GT has no class 2, model emits a few voxels."""
    ref = np.zeros((8, 8, 4), np.int16)
    ref[2:6, 2:6, :] = 1                      # class 1 only; no class 2 anywhere
    pred = ref.copy()
    pred[0, 0, 0] = 2                         # a handful of spurious class-2 voxels

    skipped = multiclass_dice(pred, ref, NC)
    counted = multiclass_dice(pred, ref, NC, skip_absent_in_ref=False)
    assert skipped > 0.9, "class 1 is near-perfect; absent class 2 must not drag it down"
    assert counted < skipped, "the old behaviour averaged in a 0 for the absent class"


def test_hallucinated_classes_reports_what_dice_now_skips():
    ref = np.zeros((8, 8, 4), np.int16)
    ref[2:6, 2:6, :] = 1
    pred = ref.copy()
    pred[0, 0, 0] = 2
    h = hallucinated_classes(pred, ref, NC)
    assert h == {2: 1}, "spurious classes must still be surfaced, just not via Dice"


def test_no_hallucination_reported_when_clean():
    ref = np.zeros((8, 8, 4), np.int16)
    ref[2:6, 2:6, :] = 1
    assert hallucinated_classes(ref.copy(), ref, NC) == {}


def test_perfect_and_disjoint_extremes():
    ref = np.zeros((8, 8, 4), np.int16)
    ref[2:6, 2:6, :] = 1
    assert multiclass_dice(ref.copy(), ref, NC) == pytest.approx(1.0)
    assert dice(np.zeros_like(ref), ref) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# zone_dice: the apex/base instrument
# --------------------------------------------------------------------------- #
def test_zone_dice_exposes_collapsed_extremes():
    """A model emitting mid-sized structures everywhere must show apex,base << mid."""
    gt = _tapered_volume()
    pred = np.zeros_like(gt)
    yy, xx = np.ogrid[:H, :W]
    for k in range(D):                        # always mid-sized, ignoring position
        pred[..., k][(yy - 16) ** 2 + (xx - 16) ** 2 <= 12 * 12] = 1
        pred[..., k][(yy - 16) ** 2 + (xx - 16) ** 2 <= 5 * 5] = 2
    z = zone_dice(pred, gt, NC)
    assert z["apex"] < z["mid"] and z["base"] < z["mid"]


def test_zone_dice_is_uniform_for_a_perfect_reconstruction():
    gt = _tapered_volume()
    z = zone_dice(gt.copy(), gt, NC)
    assert z["apex"] == pytest.approx(1.0)
    assert z["mid"] == pytest.approx(1.0)
    assert z["base"] == pytest.approx(1.0)


def test_zone_dice_skips_slices_empty_in_both():
    """Padding at the ends must not inflate apex/base with free 1.0 scores."""
    gt = np.zeros((H, W, 6), np.int16)
    gt[10:22, 10:22, 2:4] = 1                 # foreground only in the middle
    pred = gt.copy()
    pred[10:22, 10:22, 2] = 0                 # miss one of the two real slices
    z = zone_dice(pred, gt, 2)
    assert np.isnan(z["apex"]) and np.isnan(z["base"]), "empty thirds must be nan, not 1.0"
    assert z["mid"] < 1.0


def test_zone_dice_handles_empty_volume():
    z = zone_dice(np.zeros((4, 4, 0), np.int16), np.zeros((4, 4, 0), np.int16), 2)
    assert all(np.isnan(v) for v in z.values())


# --------------------------------------------------------------------------- #
# corrupt_volume: must actually span the quality range
# --------------------------------------------------------------------------- #
def test_severity_zero_is_the_identity():
    gt = torch.zeros(4, 1, 16, 16)
    gt[:, :, 4:12, 4:12] = 1
    assert torch.equal(corrupt_volume(gt, 1, 0.0, seed=0), gt)


@pytest.mark.slow
def test_severity_degrades_monotonically_and_reaches_bad_masks():
    """The flaw that made QC calibration unmeasurable: everything scored 0.81+."""
    gt_np = _tapered_volume()
    gt = torch.from_numpy(gt_np.transpose(2, 0, 1)).unsqueeze(1).float()
    scores = []
    for sev in (0.25, 0.5, 1.0):
        c = corrupt_volume(gt, NC, sev, seed=0)
        cn = c[:, 0].numpy().transpose(1, 2, 0)
        scores.append(multiclass_dice(cn, gt_np, NC))
    assert scores[0] > scores[-1], "higher severity must degrade the candidate more"
    assert scores[-1] < 0.75, (
        f"severity 1.0 only reached Dice {scores[-1]:.2f}; without genuinely bad "
        "candidates the QC-vs-true correlation is undefined rather than poor")


def test_dropout_is_biased_toward_the_ends():
    """Slice loss should concentrate at apex/base, where segmentors really fail."""
    gt = torch.ones(21, 1, 8, 8)
    lost = np.zeros(21)
    for seed in range(40):
        c = corrupt_volume(gt, 1, 1.0, seed=seed)
        lost += ((c[:, 0] > 0).sum(dim=(1, 2)) == 0).numpy()
    ends = lost[:7].sum() + lost[14:].sum()
    middle = lost[7:14].sum()
    assert ends > middle, f"dropout not end-biased (ends={ends}, mid={middle})"


# --------------------------------------------------------------------------- #
# Corruption curriculum (nnqc.corruptions.scaled_config + train._sample_corruption_cfg)
# --------------------------------------------------------------------------- #
def test_scaled_config_is_identity_at_one():
    from nnqc.corruptions import DEFAULT_CFG, scaled_config
    assert scaled_config(1.0) == DEFAULT_CFG


def test_scaled_config_scales_probs_and_magnitudes():
    from nnqc.corruptions import DEFAULT_CFG, scaled_config
    half = scaled_config(0.5)
    assert half["elastic_prob"] == pytest.approx(DEFAULT_CFG["elastic_prob"] * 0.5)
    assert half["erosion_fraction"][1] == pytest.approx(DEFAULT_CFG["erosion_fraction"][1] * 0.5)
    assert half["blob_num"] == DEFAULT_CFG["blob_num"], "counts are not magnitudes"
    assert half["max_operations"] >= 1, "must always apply at least one op when corrupting"


def test_disabled_curriculum_does_not_touch_the_rng():
    """A 'disabled' curriculum must not perturb the corruption stream.

    `_sample_corruption_cfg` draws from the global `random` module. If it drew
    even when the range is degenerate, every downstream corruption decision
    would shift - so a run resumed under the new code would diverge from the run
    it resumes, despite the setting being off.
    """
    import random

    from nnqc.train import _sample_corruption_cfg
    random.seed(11)
    before = random.random()
    random.seed(11)
    _sample_corruption_cfg(1.0, 1.0)
    assert random.random() == before, "degenerate range consumed a random variate"


def test_enabled_curriculum_varies_and_spans_the_range():
    import random

    from nnqc.train import _sample_corruption_cfg
    random.seed(3)
    scales = [_sample_corruption_cfg(0.0, 2.0)[1] for _ in range(200)]
    assert min(scales) < 0.3 and max(scales) > 1.7, "should reach both ends"
    assert len(set(scales)) > 100, "should vary per call"


def test_curriculum_reproduces_legacy_corruption_bitwise():
    """Guards the claim that the default setting changes nothing."""
    import random

    import torch

    from nnqc.corruptions import corrupt_ohe_masks_v2
    from nnqc.train import _sample_corruption_cfg

    m = torch.zeros(2, 3, 32, 32)
    m[:, 0] = 1
    m[:, 1, 8:20, 8:20] = 1
    m[:, 0, 8:20, 8:20] = 0

    def seed():                      # corrupt_binary_2d samples from np.random
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    seed()
    legacy = corrupt_ohe_masks_v2(m.clone(), corruption_prob=1.0)
    seed()
    cfg, _ = _sample_corruption_cfg(1.0, 1.0)
    current = corrupt_ohe_masks_v2(m.clone(), corruption_prob=1.0, config=cfg)
    assert torch.equal(legacy, current)


def test_validation_is_pinned_to_a_single_severity():
    """`val_loss` selects the best checkpoint, so it must not move with the curriculum.

    If validation sampled `U(lo, hi)` like training does, each epoch's val_loss
    would be drawn from a different distribution, and the `ema_val < best_val`
    comparison - where `best_val` is restored from `diffusion_best_val.txt` and
    was measured at severity 1.0 - would be meaningless.
    """
    import inspect

    from nnqc import train as T
    src = inspect.getsource(T._run_diffusion)
    val_src = src[src.index("def compute_val_loss"):]
    val_src = val_src[: val_src.index("return val_loss_sum")]
    # The invariant is FIXED severities: a degenerate (s, s) call per grid
    # entry never draws a variate, so val_loss stays comparable across epochs.
    assert "_sample_corruption_cfg(_vsev, _vsev)" in val_src, \
        "validation must use fixed severities from the val grid"
    assert "_sample_corruption_cfg(corr_lo, corr_hi)" not in val_src, \
        "validation must not sample the training curriculum"
    assert 'dt.get("val_severities", [1.0])' in src, \
        "the grid must default to [1.0] - the bit-compatible historical pass"


def test_changed_corruption_range_invalidates_the_best_val_sidecar():
    """A widened range must not inherit the previous regime's `best_val`.

    Validation is pinned to severity 1.0, so a model trained on U(0, 2) - a
    generalist - usually scores worse there than the severity-1.0 specialist
    that set the stored best. Carrying `best_val` across the change means
    `ema_val < best_val` never fires again: training burns days and
    `checkpoint="best"` silently keeps returning the pre-change model.
    """
    import inspect

    from nnqc import train as T
    src = inspect.getsource(T._run_diffusion)
    assert "diffusion_corruption_range.txt" in src, "the regime must be recorded"
    assert "prev_range != cur_range" in src, "a changed regime must be detected"
    # and the superseded best must be kept, not overwritten
    assert "_sev{_tag}.pt" in src, "the previous best checkpoint must be preserved"


def test_best_checkpoint_is_selected_on_the_reconstruction_term():
    """The total loss picked cardiac epoch 91 over weights 62% better.

    `val_diffusion_loss` = noise MSE + lambda_recon * Dice, dominated by the MSE
    term over randomly sampled timesteps. Measured on the trained cardiac model:
    `best` (epoch 91, chosen by total loss) reconstructs at 0.369 Dice while
    `_last` (epoch ~1990) reaches 0.598. The selector scored 1900 epochs of real
    improvement as worse, so it must default to the reconstruction term.
    """
    import inspect

    from nnqc import train as T
    src = inspect.getsource(T._run_diffusion)
    assert 'dt.get("best_metric", "recon")' in src, "must default to the recon term"
    assert "selector < best_val" in src, "selection must use the chosen metric"
    # and the two scalars live on different scales, so switching must invalidate
    assert 'f"{corr_lo},{corr_hi}|{best_metric}"' in src, \
        "best_metric must be part of the regime the best_val sidecar is keyed on"


def test_validation_components_are_averaged_not_last_batch():
    """A one-batch, rank-0-only number is too noisy to select checkpoints with."""
    import inspect

    from nnqc import train as T
    src = inspect.getsource(T._run_diffusion)
    val_src = src[src.index("def compute_val_loss"):]
    val_src = val_src[: val_src.index("for epoch in range")]
    assert "vl2_b = vl2_b + vl2.detach()" in val_src, "l2 must accumulate over severities"
    assert "l2_sum = l2_sum + vl2_b / k" in val_src, "and over batches, severity-averaged"
    assert "l2_sum = l2_sum / max(n, 1)" in val_src, "l2 must be averaged"
    assert "for _t in (val_loss_sum, l1_sum, l2_sum)" in val_src, "all three must all-reduce"
    # match the assignment, not the word - it appears in the explanatory comment
    assert "last_l2 = vl2" not in val_src, "the last-batch-only assignment must be gone"
    assert "last_l1 = vl1" not in val_src, "same for the noise term"
