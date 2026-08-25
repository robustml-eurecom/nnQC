"""Tests for the label conversion around the frozen MAISI mask VAE.

The autoencoder itself needs a 16 GB bundle and a GPU, but the part most likely
to be silently wrong is the *label bookkeeping*: nnQC task indices -> MAISI ids
-> bit planes -> 125-channel logits -> back to task indices. That is pure tensor
logic and is tested here on CPU.
"""
from __future__ import annotations

import torch

from nnqc.maisi_vae import TASK_LABEL_MAPS, TASK_MAP_QUALITY, MaisiMaskVAE, binarize_labels


# --------------------------------------------------------------------------- #
# Bit-plane encoding
# --------------------------------------------------------------------------- #
def test_binarize_labels_shape_and_bits():
    x = torch.zeros(1, 1, 2, 2, 2, dtype=torch.uint8)
    x[0, 0, 0, 0, 0] = 1        # 0b00000001
    x[0, 0, 0, 0, 1] = 26       # 0b00011010
    x[0, 0, 0, 1, 0] = 118      # 0b01110110
    out = binarize_labels(x)
    assert out.shape == (1, 8, 2, 2, 2)
    assert out[0, :, 0, 0, 0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert out[0, :, 0, 0, 1].tolist() == [0, 1, 0, 1, 1, 0, 0, 0]
    assert out[0, :, 0, 1, 0].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]


def test_binarize_labels_is_invertible():
    """The encoding must be lossless for every id in MAISI's taxonomy."""
    x = torch.arange(132, dtype=torch.uint8).reshape(1, 1, 132, 1, 1)
    planes = binarize_labels(x).long()
    weights = 2 ** torch.arange(8)
    recovered = (planes * weights.view(1, 8, 1, 1, 1)).sum(1)
    assert torch.equal(recovered.squeeze(), x.squeeze().long())


def test_background_encodes_to_all_zero_planes():
    x = torch.zeros(1, 1, 3, 3, 3, dtype=torch.uint8)
    assert binarize_labels(x).sum() == 0


# --------------------------------------------------------------------------- #
# Task <-> MAISI label mapping (exercised without constructing the network)
# --------------------------------------------------------------------------- #
# Real channel ids from the bundle, so the tests pin the actual numbering rather
# than the (wrong) assumption that decoder channel == label id.
BUNDLE = "/lustre/fswork/projects/rech/rpv/commun/grace-med/checkpoints/maisi/maisi_ct_generative"
CH = {1: 1, 26: 21, 118: 111, 115: 110}      # label_id -> decoder channel


class _MapOnly:
    """The mapping half of MaisiMaskVAE, without the 16 GB bundle."""

    def __init__(self, task):
        self.label_map = TASK_LABEL_MAPS[task]
        self.id_to_channel = CH

    to_maisi_labels = MaisiMaskVAE.to_maisi_labels
    from_maisi_logits = MaisiMaskVAE.from_maisi_logits


def test_decoder_channel_is_not_the_label_id():
    """Pins the trap: indexing logits by label id reads an unrelated organ.

    label_dict_124_to_132.json disagrees with label_dict.json for 123 of 125
    entries. Only background and liver coincide.
    """
    assert CH[1] == 1                      # liver: the one that coincides
    assert CH[26] != 26                    # hepatic tumor: channel 21
    assert CH[118] != 118                  # prostate: channel 111
    assert CH[115] != 115                  # heart: channel 110


def test_liver_maps_to_the_liver_label():
    """Liver is a binary whole-organ task: one foreground class -> MAISI `liver`."""
    v = _MapOnly("liver")
    labels = torch.tensor([0, 1], dtype=torch.uint8).reshape(1, 1, 2, 1, 1)
    out = v.to_maisi_labels(labels)
    assert out.flatten().tolist() == [0, 1]          # background, liver


def test_liver_round_trips_through_logits():
    v = _MapOnly("liver")
    logits = torch.full((1, 125, 2, 1, 1), -10.0)
    logits[0, 0, 0] = 10.0         # background wins voxel 0
    logits[0, CH[1], 1] = 10.0     # liver channel wins voxel 1
    out = v.from_maisi_logits(logits)
    assert out.flatten().tolist() == [0, 1]


def test_unmapped_maisi_organ_cannot_win_a_voxel():
    """Only background plus this task's own ids compete."""
    v = _MapOnly("liver")
    logits = torch.full((1, 125, 1, 1, 1), -10.0)
    logits[0, 55] = 50.0          # some unrelated organ's channel, hugely confident
    logits[0, CH[1]] = 1.0        # liver, mildly confident
    out = v.from_maisi_logits(logits)
    assert out.flatten().tolist() == [1], "an unmapped organ leaked into the output"


def test_prostate_zones_collapse_onto_one_id():
    """Documents the lossy mapping rather than pretending it round-trips."""
    v = _MapOnly("prostate")
    labels = torch.tensor([1, 2], dtype=torch.uint8).reshape(1, 1, 2, 1, 1)
    assert v.to_maisi_labels(labels).flatten().tolist() == [118, 118]

    # Both task classes point at the same channel, so decoding cannot tell them
    # apart - the tie resolves to the lower task index, by construction.
    logits = torch.full((1, 125, 2, 1, 1), -10.0)
    logits[0, CH[118]] = 10.0
    assert set(v.from_maisi_logits(logits).flatten().tolist()) == {1}


def test_cardiac_classes_all_collapse_onto_heart():
    v = _MapOnly("cardiac")
    labels = torch.tensor([1, 2, 3], dtype=torch.uint8).reshape(1, 1, 3, 1, 1)
    assert v.to_maisi_labels(labels).flatten().tolist() == [115, 115, 115]


def test_map_quality_is_recorded_for_every_task():
    assert set(TASK_LABEL_MAPS) == set(TASK_MAP_QUALITY)
    assert TASK_MAP_QUALITY["liver"] == "exact"
    for t in ("prostate", "cardiac"):
        assert TASK_MAP_QUALITY[t] != "exact", f"{t} must not claim an exact mapping"


# --------------------------------------------------------------------------- #
# Padding helpers (MAISI downsamples by 4 in all three axes)
# --------------------------------------------------------------------------- #
def test_pad_to_multiple_and_crop_round_trip():
    vol = torch.rand(1, 1, 250, 251, 37)
    padded, shape = MaisiMaskVAE.pad_to_multiple(vol, 4)
    assert all(s % 4 == 0 for s in padded.shape[-3:])
    assert shape == (250, 251, 37)
    assert torch.equal(MaisiMaskVAE.crop_to(padded, shape), vol)


def test_pad_is_a_noop_when_already_aligned():
    vol = torch.rand(1, 1, 256, 256, 32)
    padded, shape = MaisiMaskVAE.pad_to_multiple(vol, 4)
    assert padded.shape == vol.shape and shape == (256, 256, 32)
