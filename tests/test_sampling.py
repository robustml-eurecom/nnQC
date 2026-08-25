"""Tests for the sampling loop in :func:`nnqc.refine.sample_probabilities`.

The real loop needs an autoencoder, a UNet, UniMedCLIP and a DDIM scheduler on a
GPU. The *logic* around them - how ensemble draws are combined, and that the
output is a valid probability map - is independent of all that, so it is
exercised here against minimal fakes on CPU.
"""
from __future__ import annotations

import torch

from nnqc.refine import sample_probabilities

N, C, H, W = 2, 3, 8, 8
LATENT = (N, 2, 4, 4)


class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([2, 1, 0])

    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1)

    def step(self, eps, t, z):
        return (z * 0.5,)


class FakeAutoencoder:
    def __init__(self, winning_class=1):
        self.winning_class = winning_class
        self.decode_calls = 0

    def encode_stage_2_inputs(self, x):
        return torch.zeros(LATENT)

    def decode_stage_2_outputs(self, z):
        self.decode_calls += 1
        out = torch.zeros(N, C, H, W)
        out[:, self.winning_class] = 5.0
        return out


def fake_unet(x, timesteps=None, context=None):
    return torch.zeros(LATENT)


class FakeXA:
    """Stands in for CLIPCrossAttentionGrid: returns a 2-token context."""

    def build_context(self, scans, ext_features, mask=None):
        b = scans.shape[0]
        scan_token = torch.zeros(b, 512)
        pos_token = ext_features if ext_features.dim() == 2 else ext_features.squeeze(1)
        return torch.stack([scan_token, pos_token], dim=1)      # [B, 2, 512]


fake_xa = FakeXA()


def fake_embed(ratios):
    return torch.zeros(ratios.shape[0], 512)


def _inputs():
    scans = torch.rand(N, 1, H, W)
    labels = torch.zeros(N, 1, H, W)
    labels[:, :, 2:6, 2:6] = 1.0
    ratios = torch.tensor([[0.0], [1.0]])
    return scans, labels, ratios


def _run(ae, *, num_samples=1, num_classes=C, num_steps=3):
    scans, labels, ratios = _inputs()
    return sample_probabilities(
        autoencoder=ae, unet=fake_unet, xa=fake_xa, embed=fake_embed,
        scheduler=FakeScheduler(), scale_factor=1.0,
        scans=scans, labels=labels, ratios=ratios,
        num_classes=num_classes, latent_shape=LATENT,
        num_steps=num_steps, num_samples=num_samples,
        seed=0, device=torch.device("cpu"), autocast_enabled=False,
    )


def test_returns_normalised_probabilities():
    prob = _run(FakeAutoencoder())
    assert prob.shape == (N, C, H, W)
    assert torch.allclose(prob.sum(1), torch.ones(N, H, W), atol=1e-5)
    assert prob.argmax(1).eq(1).all()


def test_ensembling_runs_one_decode_per_draw():
    ae = FakeAutoencoder()
    _run(ae, num_samples=4)
    assert ae.decode_calls == 4


def test_single_draw_runs_once():
    ae = FakeAutoencoder()
    _run(ae, num_samples=1)
    assert ae.decode_calls == 1


def test_ensemble_average_is_the_mean_of_draws():
    """With a deterministic decoder every draw is identical, so the mean must be too."""
    one = _run(FakeAutoencoder(), num_samples=1)
    many = _run(FakeAutoencoder(), num_samples=5)
    assert torch.allclose(one, many, atol=1e-6)


def test_ensemble_averages_disagreeing_draws():
    """Two decoders favouring different classes must average, not winner-take-all."""

    class Alternating(FakeAutoencoder):
        def decode_stage_2_outputs(self, z):
            out = torch.zeros(N, C, H, W)
            out[:, 1 if self.decode_calls % 2 == 0 else 2] = 5.0
            self.decode_calls += 1
            return out

    prob = _run(Alternating(), num_samples=2)
    assert torch.allclose(prob[:, 1], prob[:, 2], atol=1e-5)


def test_binary_task_returns_single_channel():
    class BinaryAE(FakeAutoencoder):
        def decode_stage_2_outputs(self, z):
            self.decode_calls += 1
            return torch.full((N, 1, H, W), 3.0)

    prob = _run(BinaryAE(), num_samples=2, num_classes=1)
    assert prob.shape == (N, 1, H, W)
    assert (prob > 0.5).all()          # sigmoid(3) ~ 0.95


def test_check_reads_per_task_num_steps_from_config():
    """The step optimum is anatomy-dependent (cardiac: 2, prostate/liver: 5).

    `check(num_steps=None)` must resolve the task's measured optimum from the
    config's `inference.num_steps`, so a caller who just says `task="cardiac"`
    gets the right chain length without knowing the sweep exists.
    """
    import inspect

    from nnqc import infer as I
    src = inspect.getsource(I.check)
    assert "num_steps: int | None = None" in src, "default must be None, not a number"
    assert 'getattr(cfg, "inference", {}).get("num_steps", 5)' in src, \
        "None must resolve config -> 5, in that order"


def test_cardiac_config_carries_its_measured_step_optimum():
    import json
    from pathlib import Path
    d = json.loads(Path("configs/jz/cardiac/config.json").read_text())
    assert d.get("inference", {}).get("num_steps") == 2, \
        "cardiac's measured optimum (results/cardiac_wide_last_sweep.json) is 2"
