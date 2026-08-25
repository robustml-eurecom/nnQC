"""Tests for the post-processing helpers in nnqc.refine.

These run on CPU with synthetic volumes - no GPU, no checkpoints - so they are
safe to run on a login node::

    pytest tests/test_refine.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from nnqc.refine import (
    PRESETS,
    PostprocessConfig,
    fill_holes_per_slice,
    keep_largest_component,
    postprocess_volume,
    smooth_along_z,
)

H = W = 32
D = 8
C = 3


def dice(a, b, c):
    a, b = a == c, b == c
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * (a & b).sum() / denom


@pytest.fixture
def volumes():
    """A thick class 1 with a *thin* class 2 inside it, plus typical 2-D artefacts."""
    gt = np.zeros((H, W, D), np.int16)
    gt[8:24, 8:24, 1:7] = 1          # thick: 6 slices
    gt[12:18, 12:18, 2:5] = 2        # thin: 3 slices, nested inside class 1

    noisy = gt.copy()
    noisy[2:5, 2:5, 3] = 1           # isolated off-organ blob on one slice
    noisy[14:16, 14:16, 4] = 0       # punched hole inside class 2

    prob = np.stack([(noisy == c).astype(np.float32) for c in range(C)])
    prob += 0.01 * np.random.RandomState(0).rand(*prob.shape)
    return gt, noisy, prob


def test_largest_component_removes_isolated_blob(volumes):
    gt, _, prob = volumes
    raw = postprocess_volume(prob.copy(), C, PostprocessConfig())
    assert ((raw == 1) & (gt == 0)).sum() > 0, "fixture should contain a stray blob"

    cleaned = postprocess_volume(prob.copy(), C, PostprocessConfig(largest_component=True))
    assert ((cleaned == 1) & (gt == 0)).sum() == 0
    assert dice(cleaned, gt, 1) >= dice(raw, gt, 1)


def test_fill_holes_recovers_punched_hole(volumes):
    gt, _, prob = volumes
    raw = postprocess_volume(prob.copy(), C, PostprocessConfig())
    filled = postprocess_volume(prob.copy(), C, PostprocessConfig(fill_holes=True))
    assert dice(filled, gt, 2) >= dice(raw, gt, 2)


def test_default_cleanup_never_hurts(volumes):
    """The preset clean-up must improve, or at worst preserve, every class."""
    gt, _, prob = volumes
    raw = postprocess_volume(prob.copy(), C, PostprocessConfig())
    cleaned = postprocess_volume(prob.copy(), C, PRESETS["cleanup"]["postprocess"])
    for c in (1, 2):
        assert dice(cleaned, gt, c) >= dice(raw, gt, c) - 1e-9, f"class {c} regressed"
    assert set(np.unique(cleaned)) <= set(range(C))


def test_fill_holes_never_overwrites_a_nested_class():
    """The bug that made the default clean-up erase a whole label.

    Class 2 nested inside a class-1 ring is topologically a *hole* in class 1.
    An unguarded binary_fill_holes on class 1 therefore relabels the entire
    nested structure as class 1 - which is exactly the ACDC myocardium/LV
    geometry, so it would have silently destroyed the blood pool.
    """
    labels = np.zeros((H, W, D), np.int16)
    labels[8:24, 8:24, 1:6] = 1        # thick ring-forming class
    labels[12:20, 12:20, 1:6] = 2      # nested class, a "hole" in class 1

    out = fill_holes_per_slice(labels.copy(), num_classes=3)
    assert (out == 2).sum() == (labels == 2).sum(), "nested class was swallowed"
    assert np.array_equal(out, labels)


def test_z_smoothing_is_mild_on_this_geometry(volumes):
    """z-smoothing at sigma=0.7 is not, on its own, destructive here.

    Kept as an explicit measurement rather than an assumption: the earlier
    collapse of class 2 traced to unguarded hole-filling, not to smoothing. It
    still stays out of the shipped presets until a per-task sweep justifies it,
    because its risk profile depends on slice thickness.
    """
    gt, _, prob = volumes
    raw = postprocess_volume(prob.copy(), C, PostprocessConfig())
    smoothed = postprocess_volume(prob.copy(), C, PostprocessConfig(z_sigma=0.7))
    assert dice(smoothed, gt, 2) >= dice(raw, gt, 2) - 0.05
    assert dice(smoothed, gt, 1) >= dice(raw, gt, 1) - 0.05


def test_no_shipped_preset_enables_z_smoothing_by_default():
    for name in ("baseline", "cleanup", "ensemble", "long_chain"):
        assert PRESETS[name]["postprocess"].z_sigma == 0.0, name
    # The opt-in arm is allowed to, and exists precisely to be measured.
    assert PRESETS["z_smooth"]["postprocess"].z_sigma > 0.0


def test_smooth_along_z_shape_and_noop():
    prob = np.random.RandomState(1).rand(C, H, W, D).astype(np.float32)
    assert smooth_along_z(prob, 0.0) is prob            # disabled is a pass-through
    assert smooth_along_z(prob, 0.7).shape == prob.shape


def test_keep_largest_component_preserves_single_component():
    labels = np.zeros((H, W, D), np.int16)
    labels[8:16, 8:16, 2:5] = 1
    out = keep_largest_component(labels.copy(), num_classes=2)
    assert np.array_equal(out, labels)


def test_fill_holes_is_idempotent():
    labels = np.zeros((H, W, D), np.int16)
    labels[8:24, 8:24, 1:5] = 1
    labels[14:16, 14:16, 2] = 0
    once = fill_holes_per_slice(labels.copy(), num_classes=2)
    twice = fill_holes_per_slice(once.copy(), num_classes=2)
    assert np.array_equal(once, twice)


def test_binary_case_runs():
    prob = np.zeros((1, H, W, D), np.float32)
    prob[0, 8:24, 8:24, 1:6] = 1.0
    prob[0, 2:4, 2:4, 3] = 1.0        # stray blob
    out = postprocess_volume(prob, 1, PostprocessConfig(largest_component=True))
    assert out[2:4, 2:4, 3].sum() == 0
    assert out[8:24, 8:24, 1:6].all()


# --------------------------------------------------------------------------- #
# The apex/base amputation the benchmark caught
# --------------------------------------------------------------------------- #
def test_largest_component_amputates_disconnected_apex():
    """Why no shipped preset uses `largest_component`.

    At apex and base the cross-section is small, and one missed slice
    disconnects it from the organ body in 3-D. `keep_largest_component` then
    deletes real anatomy. Measured on prostate: apex -0.10, base -0.16 Dice, with
    the base structure erased outright in 8 of 25 volume/severity pairs.
    """
    labels = np.zeros((H, W, D), np.int16)
    labels[10:22, 10:22, 3:6] = 1        # main body
    labels[14:18, 14:18, 0] = 1          # apex island, separated by an empty slice

    amputated = keep_largest_component(labels.copy(), num_classes=2, largest_only=True)
    assert (amputated[..., 0] == 1).sum() == 0, "fixture should lose the apex island"

    # A fraction-of-largest threshold does not rescue it either: the island is
    # 16 voxels against a 432-voxel body (<4%), so any sane min_frac deletes it
    # too. That is why the shipped default does no component filtering at all.
    kept = keep_largest_component(labels.copy(), num_classes=2,
                                  min_frac=0.10, largest_only=False)
    assert (kept[..., 0] == 1).sum() == 0


def test_min_frac_removes_specks_when_explicitly_enabled():
    labels = np.zeros((H, W, D), np.int16)
    labels[10:22, 10:22, 3:6] = 1        # main body: 432 voxels
    labels[0, 0, 0] = 1                  # 1-voxel speck, well under 10%
    out = keep_largest_component(labels.copy(), num_classes=2,
                                 min_frac=0.10, largest_only=False)
    assert out[0, 0, 0] == 0, "tiny isolated specks should still go"
    assert (out[10:22, 10:22, 3:6] == 1).all(), "the organ body must survive"


def test_shipped_cleanup_preset_does_not_amputate():
    """Regression on the preset itself, not just the primitive."""
    prob = np.zeros((2, H, W, D), np.float32)
    body = np.zeros((H, W, D), bool)
    body[10:22, 10:22, 3:6] = True
    apex = np.zeros((H, W, D), bool)
    apex[14:18, 14:18, 0] = True
    prob[1][body | apex] = 1.0
    prob[0][~(body | apex)] = 1.0
    out = postprocess_volume(prob, 2, PRESETS["cleanup"]["postprocess"])
    assert out[..., 0].sum() > 0, "the cleanup preset must not delete apex structure"
