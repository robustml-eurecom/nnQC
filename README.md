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

(Plain `pip install -e .` also works.)

The package depends on PyTorch with CUDA. If you need a specific CUDA
version, install `torch` first from the official wheel index, then
`pip install -e .` will pick up the rest.

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

1. **Autoencoder** (`scripts/train_autoencoder.py`) - encodes/decodes
   one-hot segmentation masks into a low-dimensional latent.
2. **Diffusion UNet** (`scripts/train_diffusion.py`) - denoises the mask
   latent, conditioned on (a) the corrupted mask resized to latent
   resolution, (b) CLIP image features of the scan, and (c) a slice-ratio
   embedding.

Both stages are configured by **two JSON files**:

- `-e <env.json>` - paths and dataset settings (model dir, task name,
  modality, num_classes, image/label glob patterns, resume options).
- `-c <config.json>` - network architecture, training hyper-parameters,
  noise scheduler settings.

Two reference task configurations ship with the repo:

| Task | num_classes | Modality | Config dir |
|------|-------------|----------|------------|
| MSD Prostate (apex / TZ / PZ) | 3 | MRI T2 | `configs/prostate/` |
| MSD Spleen | 1 | CT | `configs/spleen/` |

---

## Quickstart

### 1. Train the autoencoder

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_autoencoder.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    -g 1
```

`-g` is the **number of GPUs** (use `torchrun --nproc_per_node=N ... -g N`
for multi-GPU). The autoencoder is saved as `<model_dir>/autoencoder.pt`.

### 2. Train the diffusion UNet

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_diffusion.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    -g 1
```

The diffusion script requires `<model_dir>/autoencoder.pt` to exist.
EMA-smoothed UNet weights are written to `diffusion_unet.pt` (best by val
loss) and `diffusion_unet_last.pt` (latest); cross-attention and
slice-embedding weights follow the same convention.

### 3. Visualize reconstructions

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_visualize.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    --num-volumes 3 --num-steps 5
```

Writes a 3×4 PNG (scan | corrupted | GT | sampled) per volume under
`<output_dir>/eval_step_*/`. We have found **5 DDIM steps** to give the
best quality / latency trade-off - increase `--num-steps` to 20–50 for a
finer denoising schedule.

### 4. QC calibration: Dice(GT, corr) vs Dice(pgt, corr)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_qc_calibration.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    --split val --num-corruptions 5 --num-steps 5
```

For each subject in the val split, generates 5 morphological corruptions
spanning a target Dice range, runs the model, and reports:

- `Dice(GT, corruption)` - the true mask quality, treated as ground-truth signal
- `Dice(pgt, corruption)` - the **nnQC predicted quality** (the QC signal)
- **MAE / RMSE / Pearson r** between the two across all subjects

For multi-class models add `--merge-foreground-classes` to evaluate a
binary foreground-vs-background calibration.

### 5. DDIM step sweep (sanity-check sampling cost)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_ddim_steps.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json
```

Renders one panel per subject with samples at [5, 10, 20, 30, 40, 50]
DDIM steps using the **same** initial noise + corruption per row, so the
only variable is the denoising schedule.

---

## Repository layout

```
nnQC/
├── nnqc/                       Importable package
│   ├── __init__.py
│   ├── xa.py                   CLIPCrossAttentionGrid (UniMedCLIP wrapper)
│   ├── corruptions.py          Morphologically realistic mask corruptions
│   ├── utils.py                Dataloaders, transforms, helpers
│   └── visualize.py            TensorBoard image helpers
├── scripts/                    Top-level CLI entry points
│   ├── train_autoencoder.py
│   ├── train_diffusion.py
│   ├── eval_visualize.py
│   ├── eval_qc_calibration.py
│   └── sweep_ddim_steps.py
├── configs/
│   ├── prostate/{config,env}.json
│   └── spleen/{config,env}.json
├── tutorials/
│   └── TUTORIAL.md             End-to-end walkthrough
├── pyproject.toml
├── LICENSE                     (MIT)
└── README.md
```

---

## Reproducing the released results

We trained two reference checkpoints for the v0.1 release:

| Task | Train epochs | Best val Dice loss (EMA) | DDIM 5 step calibration MAE |
|------|-------------:|-------------------------:|-----------------------------:|
| MSD Prostate (3-class) | 4000 | 0.314 | 0.18 (per-class) / **0.04 (merged binary)** |
| MSD Spleen (binary)    | 4000 | - | **0.056** |

Both runs use:
- `lr = 2.5e-5` with a 20-epoch linear warm-up → cosine annealing to `1e-6`
- EMA of UNet weights (`decay=0.999`), val + checkpoint always uses EMA
- Gradient clipping to `max_norm=1.0`
- Reconstruction-Dice term activates at epoch 100 (`warmup_dice_epochs`)
- DDIM 5-step sampling for inference

See [tutorials/TUTORIAL.md](tutorials/TUTORIAL.md) for the full recipe,
including the corruption-aware training loop and how to re-run the QC
calibration.

---

## Citation

If you use nnQC, please cite the release:

```bibtex
@software{nnqc_2026,
  title  = {nnQC: Segmentation Quality Control via Latent Diffusion},
  author = {robustml-eurecom},
  year   = {2026},
  url    = {https://github.com/robustml-eurecom/nnQC},
  version = {0.1.0},
}
```

See [CITATION.cff](CITATION.cff) for a machine-readable version.

---

## License

MIT - see [LICENSE](LICENSE).
