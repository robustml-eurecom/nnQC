"""The conditioning the diffusion UNet receives must depend on slice position.

This exists because it silently did not. `CLIPCrossAttentionGrid.forward` routed
the scan embedding and the slice-ratio embedding through `CrossAttentionGrid`,
which was written for N image patches against M text tokens. But
`encode_image_or_mask` returns CLIP's *pooled* vector (N=1) and the slice
embedding is a single vector (M=1), so `column_softmax` took
``softmax(dim=1)`` over a length-1 axis - identically 1.0 - and the result
reduced to ``I_proj``, exactly independent of the slice embedding.

Consequences: no position signal reached the model, `embed` received zero
gradient, and the network could only emit the average cross-section. That shows
up as apex and base slices being reconstructed mid-sized - the failure mode
reported from real runs on prostate and heart.

These tests pin the property that actually matters (position changes the
conditioning) rather than any particular implementation of it.
"""
from __future__ import annotations

import pytest
import torch

from nnqc.xa import CrossAttentionGrid


# --------------------------------------------------------------------------- #
# The degenerate behaviour, documented so nobody reintroduces it
# --------------------------------------------------------------------------- #
def test_cross_attention_grid_is_degenerate_at_n1_m1():
    """1x1 'attention' cannot mix in the second input - this is why the bug existed."""
    torch.manual_seed(0)
    grid = CrossAttentionGrid(feature_dim_i=512, feature_dim_m=512,
                              output_dim=512, grid_reduction="column_softmax").eval()
    image = torch.randn(4, 1, 512)
    with torch.no_grad():
        a, _ = grid(image, torch.zeros(4, 1, 512))
        b, _ = grid(image, torch.ones(4, 1, 512) * 5.0)
    assert torch.allclose(a, b, atol=1e-6), (
        "CrossAttentionGrid at N=1,M=1 is expected to ignore its second input; "
        "if this now fails the module changed and build_context may be revisitable"
    )


def test_cross_attention_grid_does_mix_when_given_real_sequences():
    """With N>1 the module behaves as designed - the flaw was the N=1 input."""
    torch.manual_seed(0)
    grid = CrossAttentionGrid(feature_dim_i=512, feature_dim_m=512,
                              output_dim=512, grid_reduction="column_softmax").eval()
    image = torch.randn(4, 8, 512)          # 8 patch tokens
    with torch.no_grad():
        a, _ = grid(image, torch.zeros(4, 1, 512))
        b, _ = grid(image, torch.randn(4, 1, 512) * 5.0)
    assert not torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# The property build_context must guarantee
# --------------------------------------------------------------------------- #
class _StubXA:
    """`build_context` with the CLIP tower stubbed out (no 2.3 GB backbone needed)."""

    def __init__(self, dim=512):
        torch.manual_seed(0)
        self.cross_attention = CrossAttentionGrid(
            feature_dim_i=dim, feature_dim_m=dim, output_dim=dim,
            grid_reduction="column_softmax").eval()
        self._dim = dim

    def encode_image_or_mask(self, image, is_mask=False):
        # Deterministic stand-in for the pooled CLIP embedding: [B, 1, D]
        feat = image.flatten(1)[:, : self._dim]
        if feat.shape[1] < self._dim:
            feat = torch.nn.functional.pad(feat, (0, self._dim - feat.shape[1]))
        return feat.unsqueeze(1), None

    build_context = __import__("nnqc.xa", fromlist=["CLIPCrossAttentionGrid"]) \
        .CLIPCrossAttentionGrid.build_context


@pytest.fixture
def stub():
    return _StubXA()


def test_context_has_two_tokens(stub):
    scans = torch.rand(4, 1, 32, 32)
    emb = torch.randn(4, 512)
    with torch.no_grad():
        ctx = stub.build_context(scans, emb)
    assert ctx.shape == (4, 2, 512), "context must be (scan token, position token)"


def test_context_depends_on_slice_position(stub):
    """The regression this whole file exists for."""
    scans = torch.rand(4, 1, 32, 32)
    apex = torch.zeros(4, 512)
    base = torch.ones(4, 512) * 3.0
    with torch.no_grad():
        c_apex = stub.build_context(scans, apex)
        c_base = stub.build_context(scans, base)
    assert not torch.allclose(c_apex, c_base), (
        "the slice-position embedding is not reaching the conditioning - the model "
        "cannot distinguish apex from base and will emit average-sized structures"
    )
    # The scan token is shared; only the position token should differ.
    assert torch.allclose(c_apex[:, 0], c_base[:, 0], atol=1e-6)
    assert not torch.allclose(c_apex[:, 1], c_base[:, 1])


def test_context_depends_on_the_scan(stub):
    emb = torch.randn(4, 512)
    with torch.no_grad():
        a = stub.build_context(torch.rand(4, 1, 32, 32), emb)
        b = stub.build_context(torch.rand(4, 1, 32, 32) * 10, emb)
    assert not torch.allclose(a[:, 0], b[:, 0]), "scan token must depend on the scan"


def test_position_token_carries_the_embedding_verbatim(stub):
    """Position enters as its own token, not blended away."""
    scans = torch.rand(3, 1, 32, 32)
    emb = torch.randn(3, 512)
    with torch.no_grad():
        ctx = stub.build_context(scans, emb)
    assert torch.allclose(ctx[:, 1], emb, atol=1e-6)


def test_accepts_both_2d_and_3d_embeddings(stub):
    scans = torch.rand(2, 1, 32, 32)
    emb2 = torch.randn(2, 512)
    with torch.no_grad():
        a = stub.build_context(scans, emb2)
        b = stub.build_context(scans, emb2.unsqueeze(1))
    assert torch.allclose(a, b, atol=1e-6)


def test_dimension_mismatch_is_reported(stub):
    scans = torch.rand(2, 1, 32, 32)
    with pytest.raises(ValueError, match="must match"):
        stub.build_context(scans, torch.randn(2, 128))


def test_gradient_reaches_the_slice_embedding(stub):
    """`embed` must actually train - under the old path its gradient was zero."""
    scans = torch.rand(4, 1, 32, 32)
    embed = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.GELU(),
                                torch.nn.Linear(32, 512))
    ratios = torch.rand(4, 1)
    ctx = stub.build_context(scans, embed(ratios))
    ctx.sum().backward()
    grads = [p.grad for p in embed.parameters() if p.grad is not None]
    assert grads, "no gradient reached the slice-ratio MLP"
    assert any(g.abs().sum() > 0 for g in grads), "slice-ratio MLP gradient is all zero"


# --------------------------------------------------------------------------- #
# Mask-state gate (opt-in third context token)
# --------------------------------------------------------------------------- #
def _gated(stub, dim=512):
    torch.manual_seed(1)
    stub.mask_state = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.GELU(), torch.nn.Linear(32, dim)).eval()
    return stub


def test_gate_off_ignores_mask_and_is_unchanged(stub):
    """Passing a mask to an ungated XA must be a strict no-op.

    This is what makes it safe for every call site to pass `mask=`
    unconditionally - including the training run that was already queued when
    the gate landed: gate off => identical context, identical RNG stream.
    """
    scans = torch.rand(3, 1, 32, 32)
    emb = torch.randn(3, 512)
    with torch.no_grad():
        base = stub.build_context(scans, emb)
        with_mask = stub.build_context(scans, emb, mask=torch.ones(3, 1, 32, 32))
    assert with_mask.shape == (3, 2, 512)
    assert torch.equal(base, with_mask)


def test_gate_on_appends_a_state_token(stub):
    g = _gated(stub)
    scans = torch.rand(3, 1, 32, 32)
    emb = torch.randn(3, 512)
    with torch.no_grad():
        ctx = g.build_context(scans, emb, mask=torch.ones(3, 1, 32, 32))
    assert ctx.shape == (3, 3, 512), "gate must add exactly one token"


def test_gate_distinguishes_blank_from_full_candidates(stub):
    """The point of the gate: a blank candidate must produce a different state
    token than a full one, so cross-attention CAN condition on candidate mass."""
    g = _gated(stub)
    scans = torch.rand(2, 1, 32, 32)
    emb = torch.randn(2, 512)
    blank = torch.zeros(2, 1, 32, 32)
    full = torch.ones(2, 1, 32, 32)
    with torch.no_grad():
        t_blank = g.build_context(scans, emb, mask=blank)[:, 2]
        t_full = g.build_context(scans, emb, mask=full)[:, 2]
        base = g.build_context(scans, emb, mask=blank)[:, :2]
        base2 = g.build_context(scans, emb, mask=full)[:, :2]
    assert not torch.allclose(t_blank, t_full), "state token must reflect mask mass"
    assert torch.equal(base, base2), "scan/position tokens must not depend on the mask"


def test_gated_xa_refuses_to_run_without_the_mask(stub):
    g = _gated(stub)
    with pytest.raises(ValueError, match="mask-state gate"):
        g.build_context(torch.rand(2, 1, 32, 32), torch.randn(2, 512))


def test_presets_match_shipped_weights_regime():
    """Bundled presets must match the regime of the published Zenodo weights.

    xa_mask_gate and clip_intensity change the conditioning/input
    distribution. The published checkpoints (liver, prostate, cardiac, spleen)
    were trained in the gen-2 regime (gate on, clip on), so the presets that
    load them must pin the same values. At inference the gate is auto-detected
    from the checkpoint keys (infer.py), but training resumed from a preset
    must not silently flip either flag.
    """
    import json
    from pathlib import Path
    for cfg_path in Path("nnqc/presets").glob("*/config.json"):
        if cfg_path.parent.name not in ("liver", "prostate", "cardiac", "spleen"):
            continue  # prostate_bin ships no published weights; regime-free
        d = json.loads(cfg_path.read_text())
        assert d.get("diffusion_train", {}).get("xa_mask_gate", False), cfg_path
        assert d.get("inference", {}).get("num_steps"), cfg_path
    for env_path in Path("nnqc/presets").glob("*/env.json"):
        d = json.loads(env_path.read_text())
        if d["model_dir"].endswith(("liver", "prostate", "cardiac", "spleen")):
            assert d.get("clip_intensity", False), env_path


# ---------------------------------------------------------------------------
# One-hot conditioning (in_channels == latent_channels + num_classes)
# ---------------------------------------------------------------------------

def _cfg(num_classes, in_channels, latent_channels=2):
    from argparse import Namespace
    return Namespace(num_classes=num_classes, latent_channels=latent_channels,
                     diffusion_def={"in_channels": in_channels})


def test_onehot_detection_is_structural():
    from nnqc.utils import onehot_conditioning_enabled
    assert onehot_conditioning_enabled(_cfg(4, 6))          # 2 latent + 4 classes
    assert not onehot_conditioning_enabled(_cfg(4, 3))      # legacy ordinal
    assert not onehot_conditioning_enabled(_cfg(1, 5))      # binary never one-hot


def test_onehot_condition_preserves_subcell_occupancy():
    """A class covering half of a 2x2 latent cell must read 0.5 after the
    area resize - nearest drops it to 0/1, which is exactly the class-mixing
    loss this arm removes (FINDINGS 2/14)."""
    import torch.nn.functional as F

    from nnqc.refine import _as_condition
    labels = torch.zeros(1, 1, 4, 4)
    labels[0, 0, 0, :] = 2           # class 2 covers the top row only
    cond = _as_condition(labels, 3, onehot=True)
    assert cond.shape == (1, 3, 4, 4)
    down = F.interpolate(cond, size=(2, 2), mode="area")
    assert torch.isclose(down[0, 2, 0, 0], torch.tensor(0.5))   # fractional
    near = F.interpolate(_as_condition(labels, 3), size=(2, 2), mode="nearest")
    assert near[0, 0, 0, 0] in (0.0, 2 / 3)                     # all-or-nothing


def test_onehot_condition_keeps_all_classes():
    from nnqc.refine import _as_condition
    labels = torch.tensor([[[[0, 1], [2, 3]]]]).float()
    cond = _as_condition(labels, 4, onehot=True)
    assert cond.shape == (1, 4, 2, 2)
    assert torch.allclose(cond.sum(dim=1), torch.ones(1, 2, 2))


def test_gate_fraction_needs_foreground_not_onehot_stack():
    """(ohe > 0).mean over the full one-hot stack is identically
    1/num_classes (every voxel is in exactly one class) regardless of the
    true foreground fraction, which would freeze the gate's input.
    build_context must receive a foreground mask instead."""
    from nnqc.refine import _as_condition
    labels = torch.zeros(1, 1, 8, 8)
    labels[0, 0, 2:4, 2:4] = 1                       # 4/64 foreground
    ohe = _as_condition(labels, 2, onehot=True)
    assert (ohe > 0).float().mean().item() == 0.5    # frozen at 1/2: the trap
    labels[0, 0, :4, :4] = 1                         # now 16/64 foreground
    ohe2 = _as_condition(labels, 2, onehot=True)
    assert (ohe2 > 0).float().mean().item() == 0.5   # unchanged: useless gate
    assert (labels > 0).float().mean().item() == 16 / 64   # what the gate must see
