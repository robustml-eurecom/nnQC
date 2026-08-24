"""Frozen MAISI mask autoencoder as a drop-in latent space for nnQC.

nnQC normally trains its own 2-D ``AutoencoderKL`` on one-hot masks. This module
offers the alternative: NVIDIA's MAISI **mask** autoencoder
(``mask_generation_autoencoder.pt`` from the ``maisi_ct_generative`` bundle),
frozen, so only the diffusion model has to be trained.

Three things about MAISI have to be respected or the latents are meaningless:

1. **It is 3-D.** Convolutions are ``(k, k, k)`` and it downsamples by 4 in every
   axis, so a ``256x256xD`` volume becomes a ``4 x 64 x 64 x D/4`` latent. nnQC's
   UNet is 2-D, so the LDM consumes *z-slices* of that 3-D latent - and one
   latent slice spans four image slices, which the slice-ratio conditioning has
   to account for.
2. **Masks are bit-plane encoded, not one-hot.** The encoder takes 8 channels:
   the uint8 label index written out as 8 binary planes (``binarize_labels`` in
   the MAISI bundle). We reproduce that exactly.
3. **It speaks a fixed 132-label taxonomy.** The decoder emits 125 channels
   indexed by MAISI's own ``label_dict.json``. A task's classes must be mapped
   onto those ids, and anatomy absent from the taxonomy has no faithful
   encoding - see ``TASK_LABEL_MAPS`` for what maps cleanly and what does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

# Default location of the bundle on this cluster; override via $NNQC_MAISI_DIR.
DEFAULT_BUNDLE = Path(
    "/lustre/fswork/projects/rech/rpv/commun/grace-med/checkpoints/maisi/maisi_ct_generative"
)

# nnQC task class index -> MAISI label id (from the bundle's label_dict.json).
#
# Only `liver` maps exactly. The other two are recorded here with their
# limitation stated so the gap is explicit rather than discovered later:
#   * MSD prostate is zonal (peripheral / transition); MAISI has a single
#     `prostate` label, so both zones collapse onto it.
#   * ACDC is RV / myocardium / LV; MAISI has only a whole `heart`. Encoding all
#     three onto one id destroys the class distinction entirely.
TASK_LABEL_MAPS: dict[str, dict[int, int]] = {
    "liver": {1: 1},                             # whole liver organ (binary task)
    "prostate": {1: 118, 2: 118},                # both zones -> prostate (collapses)
    "cardiac": {1: 115, 2: 115, 3: 115},         # all chambers -> heart (collapses)
}

TASK_MAP_QUALITY = {
    "liver": "exact",
    "prostate": "lossy: peripheral and transition zone both map to `prostate`",
    "cardiac": "unusable: RV/myocardium/LV all map to `heart`; MAISI has no cardiac chambers",
}


def load_id_to_channel(bundle) -> dict[int, int]:
    """Map MAISI label id -> decoder output channel.

    ``label_dict_124_to_132.json`` is ``name -> [decoder_channel, label_id]``.
    The two numberings coincide only for background and liver; for the other 123
    entries they differ, so anything that indexes the decoder's 125 channels by
    label id silently reads the wrong organ.
    """
    path = Path(bundle) / "configs" / "label_dict_124_to_132.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing - it is required to map decoder channels to label ids.")
    return {int(label_id): int(channel)
            for _name, (channel, label_id) in json.loads(path.read_text()).items()}


def binarize_labels(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Label indices ``[B, 1, H, W, D]`` -> bit planes ``[B, bits, H, W, D]``.

    Byte-for-byte the encoding MAISI trains with (bundle ``scripts/utils.py``):
    a little-endian bit decomposition of the uint8 label index.
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3)


class MaisiMaskVAE(torch.nn.Module):
    """Frozen MAISI mask autoencoder, wrapped in nnQC's label convention.

    ``encode`` takes a volume of *task* class indices and returns the MAISI
    latent; ``decode`` inverts it back to task class indices.
    """

    def __init__(self, task: str, bundle_dir=None, device=None, latent_channels: int = 4,
                 norm_float16: bool = True, autocast: bool = True):
        super().__init__()
        from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

        bundle = Path(bundle_dir or DEFAULT_BUNDLE)
        ckpt = bundle / "models" / "mask_generation_autoencoder.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"MAISI mask autoencoder not found at {ckpt}. Point --maisi-dir / "
                "$NNQC_MAISI_DIR at a maisi_ct_generative bundle."
            )
        if task not in TASK_LABEL_MAPS:
            raise ValueError(f"no MAISI label map for task {task!r}; "
                             f"known: {sorted(TASK_LABEL_MAPS)}")

        self.task = task
        self.label_map = TASK_LABEL_MAPS[task]
        self.map_quality = TASK_MAP_QUALITY[task]
        self.device = device or torch.device("cuda")
        # `norm_float16` makes MaisiGroupNorm3D emit fp16 to save memory, which
        # only type-checks if the following convolutions also run in fp16 - i.e.
        # under autocast, which is how the MAISI bundle deploys it. Running it
        # outside autocast raises
        #   RuntimeError: Input type (c10::Half) and bias type (float) should be the same
        # so the two flags travel together. Set both False for a pure-fp32 run.
        self.autocast = autocast and norm_float16

        # Architecture verbatim from configs/inference.json:mask_generation_autoencoder_def.
        self.net = AutoencoderKlMaisi(
            spatial_dims=3, in_channels=8, out_channels=125,
            latent_channels=latent_channels,
            num_channels=[32, 64, 128], num_res_blocks=[1, 2, 2],
            norm_num_groups=32, norm_eps=1e-6,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
            use_flash_attention=False, use_checkpointing=True,
            use_convtranspose=True, norm_float16=norm_float16,
            num_splits=8, dim_split=1,
        ).to(self.device)
        self.net.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True), strict=True)

        # Frozen: this whole point of this arm is to train only the LDM.
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        label_dict_path = bundle / "configs" / "label_dict.json"
        self.label_dict = json.loads(label_dict_path.read_text()) if label_dict_path.is_file() else {}

        # label id -> decoder output channel. These are NOT the same numbering:
        # label_dict_124_to_132.json holds `name -> [channel, label_id]` and they
        # disagree for 123 of 125 entries.
        self.id_to_channel = load_id_to_channel(bundle)
        missing = [i for i in self.label_map.values() if i not in self.id_to_channel]
        if missing:
            raise ValueError(f"MAISI label id(s) {missing} have no decoder channel in "
                             f"{bundle}/configs/label_dict_124_to_132.json")

    # -- label conversion ---------------------------------------------------
    def to_maisi_labels(self, task_labels: torch.Tensor) -> torch.Tensor:
        """Task class indices -> MAISI label ids (background stays 0)."""
        out = torch.zeros_like(task_labels, dtype=torch.uint8)
        for task_idx, maisi_id in self.label_map.items():
            out[task_labels == task_idx] = maisi_id
        return out

    def from_maisi_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """MAISI 125-channel logits -> task class indices.

        NB: the decoder's channel index is *not* the label id. The bundle's
        ``label_dict_124_to_132.json`` maps ``name -> [channel, label_id]`` and
        the two differ for 123 of 125 entries (hepatic tumor is channel 21 but
        id 26; prostate is channel 111 but id 118). Indexing logits by label id
        therefore reads a completely unrelated organ's channel - which is why an
        earlier version of this scored ~0 per-label even on MAISI's own data.

        Only the channels this task maps onto compete, plus background, so an
        unrelated MAISI organ can never win a voxel. Where two task classes share
        one MAISI id (prostate, cardiac) the tie is resolved toward the lower
        task index - the collapse is inherent to the mapping, not to this step.
        """
        chans = [0] + [self.id_to_channel[self.label_map[k]] for k in sorted(self.label_map)]
        subset = logits[:, chans]                     # [B, 1+K, ...]
        winner = subset.argmax(1)                     # 0 = background
        out = torch.zeros_like(winner, dtype=torch.uint8)
        for pos, task_idx in enumerate(sorted(self.label_map), start=1):
            out[winner == pos] = task_idx
        return out.unsqueeze(1)

    # -- codec --------------------------------------------------------------
    def _amp(self):
        from contextlib import nullcontext
        if not self.autocast or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.float16)

    @torch.no_grad()
    def encode(self, task_labels: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W, D]`` task labels -> ``[B, C, H/4, W/4, D/4]`` latent."""
        maisi = self.to_maisi_labels(task_labels)
        planes = binarize_labels(maisi.to(torch.uint8)).float().to(self.device)
        with self._amp():
            return self.net.encode_stage_2_inputs(planes)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent -> ``[B, 1, H, W, D]`` task class indices."""
        with self._amp():
            logits = self.net.decode_stage_2_outputs(latent)
        return self.from_maisi_logits(logits.float())

    @torch.no_grad()
    def round_trip(self, task_labels: torch.Tensor) -> torch.Tensor:
        """encode -> decode, for measuring how much the frozen VAE preserves."""
        return self.decode(self.encode(task_labels))

    @staticmethod
    def pad_to_multiple(vol: torch.Tensor, k: int = 4):
        """Pad the trailing 3 dims up to a multiple of ``k`` (2 downsamples = 4).

        Returns ``(padded, original_shape)`` so the crop can be undone.
        """
        *_, h, w, d = vol.shape
        ph, pw, pd = (-h) % k, (-w) % k, (-d) % k
        if ph or pw or pd:
            vol = torch.nn.functional.pad(vol, (0, pd, 0, pw, 0, ph))
        return vol, (h, w, d)

    @staticmethod
    def crop_to(vol: torch.Tensor, shape) -> torch.Tensor:
        h, w, d = shape
        return vol[..., :h, :w, :d]
