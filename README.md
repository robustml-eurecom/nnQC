# nnQC - Segmentation Quality Control via Latent Diffusion

**nnQC** is a quality-control model for medical image segmentation. It is a
2D latent diffusion model (LDM) that, given a CT/MR scan and a *corrupted*
segmentation mask, reconstructs what it believes the *correct* mask should
look like. The Dice score between an input mask and the model's
reconstruction is the QC signal - calibrated to track the true Dice against
the ground truth (see [tutorials/TUTORIAL.md](tutorials/TUTORIAL.md)).

```
        scan + corrupted mask ──► nnQC LDM ──► predicted "correct" mask (pgt)
                                                    │
                              Dice(pgt, corrupted) ─┴─► QC score
```

The pipeline is built on top of MONAI's `AutoencoderKL` and
`DiffusionModelUNet`, with CLIP-based scan conditioning (UniMedCLIP) and a
slice-ratio embedding so the same 2D model handles apex / mid / base slices.

---

## Installation

`nnqc` is a Python package and is best used inside a fresh virtualenv. We
recommend [`uv`](https://docs.astral.sh/uv/) for fast resolution:

```bash
git clone https://github.com/robustml-eurecom/nnQC.git
cd nnQC
uv venv
source .venv/bin/activate
uv pip install -e .
```

(Plain `pip install -e .` also works.) Installing registers the `nnqc`
command and ships the bundled task presets, so `--task prostate` works
out of the box.

The package depends on PyTorch with CUDA. If you need a specific CUDA
version, install `torch` first from the official wheel index, then
`uv pip install -e .` will pick up the rest. To run a one-off command
without activating the venv, prefix it with `uv run` (e.g.
`uv run nnqc list-tasks`).

### Pretrained weights

Trained checkpoints are **not** distributed in the git repo because they
exceed 1 GB per task. Download them separately and place under
`trained_weights/<TASK>/`. The exact URL is published on the GitHub
release page. Expected layout:

```
trained_weights/
└── MSD_Prostate_v2/
    ├── autoencoder.pt
    ├── diffusion_unet.pt          # best (EMA, by val loss)
    ├── diffusion_unet_last.pt     # latest (EMA)
    ├── xa.pt        / xa_last.pt
    ├── embed.pt     / embed_last.pt
└── unimed_clip_vit_b16.pt          # UniMedCLIP backbone
```

---

## Pipeline

nnQC is trained in **two stages**:

1. **Autoencoder** - encodes/decodes one-hot segmentation masks into a
   low-dimensional latent.
2. **Diffusion UNet** - denoises the mask latent, conditioned on (a) the
   corrupted mask resized to latent resolution, (b) CLIP image features of
   the scan, and (c) a slice-ratio embedding.

Both stages read a **config pair**, either a bundled preset (`--task`) or an
explicit JSON pair:

- **env.json** - paths and dataset settings (model dir, task name, modality,
  num_classes, image/label glob patterns, resume options).
- **config.json** - network architecture, training hyper-parameters, noise
  scheduler settings.

Any field can be overridden from the CLI or the Python API without editing
the JSON. Bundled presets:

| Task | num_classes | Modality | Preset |
|------|-------------|----------|--------|
| MSD Prostate | 3 | MRI T2 | `prostate` |
| MSD Prostate binary | 1 | MRI T2 | `prostate_bin` |
| MSD Spleen | 1 | CT | `spleen` |

Run `nnqc list-tasks` to see what is installed. The same JSON files also live
under `configs/` for you to copy and edit.

---

## Quickstart (CLI)

A single `nnqc` command with subcommands:

```bash
# 1. autoencoder
nnqc train-autoencoder --task prostate --epochs 500 --lr 5e-5 --device 0

# 2. diffusion UNet (needs <model_dir>/autoencoder.pt)
nnqc train-diffusion --task prostate --epochs 4000 --lr 2.5e-5 \
    --scheduler cosine --warmup-dice-epochs 100 --device 0

# 3. visualize reconstructions
nnqc evaluate --task prostate --num-volumes 3 --num-steps 5 --device 0
```

Use an explicit config pair instead of a preset with
`--config configs/prostate/config.json --env configs/prostate/env.json`.
Multi-GPU uses torchrun: `torchrun --nproc_per_node=2 -m nnqc.cli
train-diffusion --task prostate -g 2`.

EMA-smoothed UNet weights are written to `diffusion_unet.pt` (best by val
loss) and `diffusion_unet_last.pt` (latest); cross-attention and
slice-embedding weights follow the same convention. **5 DDIM steps** give the
best quality / latency trade-off; raise `--num-steps` to 20-50 for a finer
schedule.

## Quickstart (notebook / Python)

```python
import nnqc

nnqc.train_autoencoder(task="prostate", epochs=500, lr=5e-5, device=0)
nnqc.train_diffusion(
    task="prostate", epochs=4000, lr=2.5e-5,
    scheduler="cosine", warmup_dice_epochs=100, device=0,
)
nnqc.evaluate(task="prostate", num_volumes=3, num_steps=5, device=0)

# resume a run, or point at your own data:
nnqc.train_diffusion(task="prostate", resume=True, start_epoch=1000, epochs=4000)
nnqc.train_diffusion(
    config="configs/prostate/config.json",
    env="configs/prostate/env.json",
    data_dir="/data/my_prostate", model_dir="/data/runs/prostate",
)
```

Common overrides (CLI flag / Python kwarg): `epochs`, `lr`, `batch_size`,
`patch_size`, `val_interval`, `scheduler` (`cosine|constant|step|exponential`),
`warmup_dice_epochs`, `lambda_recon`, `ema_decay`, `num_train_timesteps`,
`model_dir`, `output_dir`, `data_dir`, `resume`, `start_epoch`.

---

## Quality control on a new mask

Once a model is trained, `check()` is the one-call QC entry point. Give it a
scan and a candidate segmentation; it preprocesses both (orientation, slice and
foreground crop, resize, intensity scaling), reconstructs the mask the model
believes is correct, and returns the Dice agreement as the QC score. The
reconstruction is mapped back onto the input volume's grid (shape + affine), so
you can save or overlay it directly.

```python
import nnqc

result = nnqc.check("scan.nii.gz", "candidate_mask.nii.gz", task="prostate")
print(result.qc_score)            # volume Dice(candidate, reconstruction); low = suspect mask
print(result.qc_score_per_class)  # per-class Dice (multi-class models)
print(result.slice_scores)        # per-slice Dice, with result.slice_ratios
result.save("reconstruction.nii.gz")   # written on the input grid
```

```bash
nnqc check --task prostate --image scan.nii.gz --mask candidate_mask.nii.gz \
    --save reconstruction.nii.gz
```

A high `qc_score` means the candidate agrees with what the model reconstructs
(likely good); a low score flags a probable segmentation error.

---

## Repository layout

```
nnQC/
├── nnqc/                       Importable package
│   ├── __init__.py             Public API: train_autoencoder/train_diffusion/evaluate
│   ├── cli.py                  `nnqc` command dispatcher
│   ├── config.py               JSON + kwargs config resolver, task presets
│   ├── train.py                Training loops (autoencoder + diffusion)
│   ├── evaluate.py             DDIM sampling + reconstruction panels
│   ├── infer.py                check(): one-call QC on a scan + mask pair
│   ├── xa.py                   CLIPCrossAttentionGrid (UniMedCLIP wrapper)
│   ├── corruptions.py          Morphologically realistic mask corruptions
│   ├── utils.py                Dataloaders, transforms, helpers
│   ├── visualize.py            TensorBoard image helpers
│   └── presets/                Bundled task configs (shipped in the wheel)
│       ├── prostate/{config,env}.json
│       ├── prostate_bin/{config,env}.json
│       └── spleen/{config,env}.json
├── configs/                    Editable copies of the presets
│   ├── prostate/{config,env}.json
│   └── spleen/{config,env}.json
├── tutorials/
│   └── TUTORIAL.md             End-to-end walkthrough
├── pyproject.toml
├── LICENSE                     (MIT)
└── README.md
```

---

## Citation

If you use nnQC, please cite:

```bibtex
@article{marciano2025diffusion,
  title={Diffusion-Based Quality Control of Medical Image Segmentations across Organs},
  author={Marcian{\`o}, Vincenzo and Chaptoukaev, Hava and Fernandez, Virginia and Cardoso, M Jorge and Ourselin, S{\'e}bastien and Antonelli, Michela and Zuluaga, Maria A},
  journal={arXiv preprint arXiv:2511.09588},
  year={2025}
}
```

See [CITATION.cff](CITATION.cff) for a machine-readable version.

---

## License

MIT - see [LICENSE](LICENSE).
