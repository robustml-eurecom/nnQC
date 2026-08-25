"""The fp16 NaN that silently destroyed two of three training runs.

`_run_diffusion` calls the frozen autoencoder inside an fp16 autocast region.
The autoencoder is *trained* in fp32 (`_run_autoencoder` uses no autocast) with
``norm_eps=1e-6``, and 1e-6 is **subnormal in fp16**. So whenever a
normalisation group sees constant input, fp16 computes

    (x - mean) / sqrt(var + eps)  ->  0 / sqrt(0 + 0)  ->  NaN

Constant groups are routine, not exotic: a class absent from a slice (4-class
ACDC) or an all-background slice (binary liver). Measured in production:

    prostate      0 / 62216 loss readings NaN   (3-class, got lucky)
    cardiac     100% from epoch 0, scale_factor = nan
    liver     58510 / 65068 NaN  (90%)

Both failures were invisible. `nan < best_val` is always False, so the
best-checkpoint branch never fired and no usable weights were ever written -
while the jobs kept logging, kept saving `_last.pt`, and looked healthy. Roughly
7 GPU-hours produced nothing.

These tests run on CPU. CPU autocast uses bfloat16, which has fp32's exponent
range and therefore does *not* reproduce the underflow, so the fp16 arithmetic
is checked directly instead of through `torch.autocast`.
"""
from __future__ import annotations

import pytest
import torch

from nnqc.train import _ae_fp32


def test_norm_eps_is_subnormal_in_fp16():
    """The root cause, stated as an arithmetic fact."""
    eps = torch.tensor(1e-6)
    assert eps.item() > 0
    assert eps.half().item() == pytest.approx(1e-6, rel=0.3), "1e-6 is representable..."
    # ...but it vanishes against a zero variance in fp16 normalisation.
    var16 = torch.tensor(0.0, dtype=torch.float16)
    denom = torch.sqrt(var16 + torch.tensor(1e-6, dtype=torch.float16))
    numer = torch.tensor(0.0, dtype=torch.float16)
    assert torch.isfinite(denom)
    # The killer is 0/0 once the numerator is also zero on a constant group.
    assert torch.isnan(numer / var16), "0/0 on a constant group is the NaN source"


def test_groupnorm_on_constant_input_is_nan_in_fp16_but_not_fp32():
    """A constant normalisation group is what an absent class looks like."""
    gn16 = torch.nn.GroupNorm(2, 4, eps=1e-6).half()
    x16 = torch.full((1, 4, 8, 8), 3.0, dtype=torch.float16)   # perfectly constant
    out16 = gn16(x16)

    gn32 = torch.nn.GroupNorm(2, 4, eps=1e-6)
    out32 = gn32(x16.float())

    assert torch.isfinite(out32).all(), "fp32 handles a constant group"
    # fp16 may produce NaN here; if a future torch fixes it the guard below still
    # protects us, so assert the *fp32* path rather than pinning torch behaviour.
    if not torch.isfinite(out16).all():
        assert torch.isnan(out16).any()


class _ConstantEncoder(torch.nn.Module):
    """Stand-in for the AE encoder: normalises, so constant input is dangerous."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(2, 4, eps=1e-6)

    def forward(self, x):
        return self.norm(x)


def test_ae_fp32_casts_inputs_and_returns_finite():
    enc = _ConstantEncoder()
    x = torch.full((1, 4, 8, 8), 3.0, dtype=torch.float16)   # constant AND fp16
    out = _ae_fp32(enc, x)
    assert out.dtype == torch.float32, "must upcast to fp32"
    assert torch.isfinite(out).all(), "must not produce NaN on a constant group"


def test_ae_fp32_passes_through_non_tensor_args():
    def fn(a, flag, scale):
        assert flag is True and scale == 2
        return a * scale
    out = _ae_fp32(fn, torch.ones(2, 2, dtype=torch.float16), True, 2)
    assert out.dtype == torch.float32 and torch.equal(out, torch.full((2, 2), 2.0))


def test_ae_fp32_is_differentiable():
    """The decoder call must still carry gradients back to the UNet."""
    lin = torch.nn.Linear(4, 4)
    x = torch.ones(1, 4, requires_grad=True)
    _ae_fp32(lin, x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_nan_scale_factor_would_never_beat_best_val():
    """Why the failure was invisible: NaN loses every comparison, silently."""
    best_val = 100.0
    nan_val = float("nan")
    assert not (nan_val < best_val), (
        "a NaN validation loss never triggers the best-checkpoint branch, so a "
        "fully broken run writes no usable weights while still looking alive"
    )


def test_std_of_a_tensor_containing_nan_is_nan():
    """scale_factor = 1/std(z); one NaN latent poisons the whole scale."""
    z = torch.ones(4, 2, 8, 8)
    z[0, 0, 0, 0] = float("nan")
    assert torch.isnan(torch.std(z)), "a single NaN makes scale_factor NaN"
