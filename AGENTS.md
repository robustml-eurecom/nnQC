# nnQC — Agent Guide

This file is written for AI coding agents. It assumes no prior knowledge of the project and documents the architecture, build/test workflow, code conventions, and operational details needed to work safely in this repository.

## Project overview

**nnQC** is a Python package for quality control of 3-D medical image segmentations. It trains a 2-D latent diffusion model (LDM) that, given a CT or MR scan and a *corrupted* segmentation mask, reconstructs the mask it believes is correct. The agreement between the input mask and the reconstruction (Dice by default) is the QC score.

- Paper: IEEE TMI, doi `10.1109/TMI.2026.3714697` (see `CITATION.cff`).
- Repository: `https://github.com/robustml-eurecom/nnQC`.
- License: MIT.

The package is built on PyTorch and MONAI (`AutoencoderKL`, `DiffusionModelUNet`, `DDIMScheduler`), uses a UniMed-CLIP fork for scan conditioning, and is shipped with bundled task presets for prostate (MSD), spleen (MSD), and cardiac (ACDC) segmentation tasks.

## Technology stack

- **Language:** Python ≥3.10.
- **Build backend:** `hatchling` (PEP 621 project metadata in `pyproject.toml`).
- **Deep-learning stack:** `torch>=2.4`, `torchvision>=0.19`, `monai[all]>=1.4`, `monai-generative>=0.2.3`.
- **Data / I/O:** `numpy`, `nibabel`, `scipy`, `scikit-image`, `pillow`, `einops`, `tqdm`.
- **Conditioning:** UniMed-CLIP fork (`open_clip_torch @ git+https://github.com/mbzuai-oryx/UniMed-CLIP.git`), `timm`, `transformers`, `opencv-python-headless`.
- **Loss / metrics:** `lpips` (PerceptualLoss backend), optional `medpy` for distance metrics (`hd95`, `assd`, …).
- **Packaging / environment:** `uv` is the preferred resolver; `pip install -e .` also works.
- **Linting:** `ruff`.
- **Testing:** `pytest`.
- **Logging / visualization:** TensorBoard; no Weights & Biases.

**Important dependency caveat:** `open_clip` must be the UniMed-CLIP fork, *not* the upstream `open-clip-torch` package. `nnqc/xa.py` calls `open_clip.get_mean_std` and passes `inmem=` and `text_encoder_name=` to `create_model_and_transforms`, which do not exist upstream.

## Repository layout

```text
nnQC/
├── nnqc/                       # Importable package
│   ├── __init__.py             # Public API; heavy imports are lazy
│   ├── cli.py                  # `nnqc` command dispatcher
│   ├── config.py               # JSON + kwargs config resolver, task presets
│   ├── train.py                # Training loops (autoencoder + diffusion)
│   ├── evaluate.py             # DDIM sampling and reconstruction panels
│   ├── infer.py                # check(): one-call QC on a scan + mask pair
│   ├── hub.py                  # download_weights(): Zenodo helper
│   ├── metrics.py              # Pluggable QC metrics (Dice, IoU, medpy adapters)
│   ├── xa.py                   # CLIPCrossAttentionGrid (UniMedCLIP wrapper)
│   ├── corruptions.py          # Anatomically realistic mask corruptions
│   ├── utils.py                # Dataloaders, transforms, helpers
│   ├── visualize.py            # TensorBoard image helpers
│   ├── refine.py               # Post-processing utilities
│   ├── mcp_server.py           # MCP stdio server exposing QC tools
│   ├── maisi_vae.py            # MAISI VAE utilities
│   └── presets/                # Bundled task configs (shipped in the wheel)
│       ├── prostate/{config,env}.json
│       ├── prostate_bin/{config,env}.json
│       └── spleen/{config,env}.json
├── configs/                    # Editable copies of the presets (top-level)
│   ├── jz/{cardiac,liver,prostate}/
│   ├── prostate/
│   └── spleen/
├── scripts/                    # Analysis, diagnostics, and utility scripts
├── slurm/                      # Slurm batch scripts and environment setup
├── tests/                      # pytest suite
├── trained_weights/            # Checkpoints (gitignored, distributed separately)
├── tutorials/                  # Jupyter notebooks and walkthroughs
├── pyproject.toml              # Project metadata, deps, build config, ruff rules
├── pytest.ini                 # pytest markers
├── uv.lock                     # Locked dependency tree
└── README.md
```

## Build and install

Recommended setup with `uv`:

```bash
git clone https://github.com/robustml-eurecom/nnQC.git
cd nnQC
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # dev extra = ruff + pytest
```

Plain `pip` works too:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Installing registers the `nnqc` console script and ships bundled presets so `--task prostate` works immediately.

### Jean Zay environment (where this repo lives)

On the Jean Zay cluster the project root is:

```text
/lustre/fsn1/projects/rech/rpv/commun/nnQC
```

The local `.venv` is a symlink to `$SCRATCH/nnqc/venv`. Do **not** use `uv run` / `uv sync` here; the venv is hand-assembled (CUDA torch from the cu128 index plus the UniMed-CLIP fork). Re-resolving from `pyproject.toml` rebuilds it incorrectly. Instead:

```bash
cd /lustre/fsn1/projects/rech/rpv/commun/nnQC
source .venv/bin/activate
```

Activation exports `NNQC_ROOT`, `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, `TMPDIR`, `NNQC_CLIP_PATH`, `NNQC_MAISI_DIR`, and clears a locked-out `HF_TOKEN`. A `sitecustomize.py` in the venv re-applies these defaults for processes that skip activation.

Additional constraints:

- `$WORK` is at its inode quota; never point caches or outputs there.
- H100 hours come from the `vnc@h100` allocation (`gpu_p6` partition). The `rpv` account has no H100 allocation and `rpv@a100`/`rpv@cpu` are over-consumed.
- Compute nodes have no outbound network. Warm any lazy-downloaded resources (e.g., `PerceptualLoss(network_type="squeeze")`, UniMed-CLIP backbone) on a login node first.

## Build / test / lint commands

Run tests (CPU-only, safe on a login node):

```bash
pytest tests/ -q                    # all tests
pytest tests/ -q -m "not slow"      # skip SDT-heavy corruption tests
```

As of the latest check, the suite collects 74 tests and the non-slow subset passes in ~30 s.

Lint:

```bash
ruff check scripts tests             # clean
ruff check nnqc                      # ~328 pre-existing findings, mostly whitespace in utils.py
ruff check nnqc --fix                # mechanically fixes most of them
```

**Do not introduce new lint findings.** `scripts/` and `tests/` must stay clean. The pre-existing findings in `nnqc/` (especially `utils.py`, which contains unused legacy code) are intentionally left untouched to avoid obscuring meaningful diffs.

Style configuration (`pyproject.toml`):

- `line-length = 100`
- `target-version = "py310"`
- Lint rules: `E`, `F`, `I`, `W`, `UP`
- `E501` (line too long) is ignored.

## Training and inference pipeline

### Two-stage training

1. **Autoencoder** (`nnqc train-autoencoder --task prostate ...`): compresses one-hot masks to a low-dimensional latent. Writes `<model_dir>/autoencoder.pt` and `discriminator.pt`, plus `_last.pt` siblings.
2. **Diffusion UNet** (`nnqc train-diffusion --task prostate ...`): denoises the mask latent conditioned on the corrupted mask, CLIP scan features, and slice-ratio embedding. Writes `<model_dir>/diffusion_unet.pt`, `xa.pt`, and `embed.pt`.

Multi-GPU training uses `torchrun` plus the `-g` flag:

```bash
torchrun --nproc_per_node=4 -m nnqc.cli train-diffusion --task prostate -g 4
```

### Config system

Every entry point accepts either:

- `--task <preset>` (reads `nnqc/presets/<task>/`), or
- `--config <path> --env <path>` (reads a `config.json` + `env.json` pair).

`env.json` holds paths and dataset settings; `config.json` holds network architecture and hyper-parameters. `nnqc.config.resolve_config` merges them into a flat namespace and routes friendly overrides (`epochs`, `lr`, `batch_size`, `scheduler`, …) into the correct nested block.

### QC inference

The one-call entry point is `nnqc.check()`:

```python
import nnqc
result = nnqc.check("scan.nii.gz", "candidate_mask.nii.gz", task="prostate")
print(result.qc_score)
result.save("reconstruction.nii.gz")
```

CLI equivalent:

```bash
nnqc check --task prostate --image scan.nii.gz --mask candidate_mask.nii.gz --save recon.nii.gz
```

The reconstruction is returned on the input volume's grid (shape + affine).

### Pretrained weights

Trained checkpoints are published on Zenodo as a single archive `nnQC_pretrained_weights.zip` (one `weight_<task>/` folder per task, cached under `trained_weights/.cache/`). Download with:

```bash
nnqc download prostate
# or
python -c "import nnqc; nnqc.download_weights('prostate')"
```

The record id is resolved from `ZENODO_RECORD` in `nnqc/hub.py`, overridable with the `NNQC_ZENODO_RECORD` (or per-task `NNQC_ZENODO_RECORD_<TASK>`) environment variable or the `--record` CLI flag. If no record id is known for a task, the downloader raises a clear error saying how to pass one. Checkpoints auto-download on first use of `check`/`evaluate` if missing. Weight files are placed under `trained_weights/<task>/`.

## Code organization and key modules

| File | Responsibility |
|------|----------------|
| `nnqc/cli.py` | `nnqc` command dispatcher; keeps heavy imports out of `list-tasks`/`--version`/`download`. |
| `nnqc/config.py` | Merges JSON configs, resolves task presets, routes friendly overrides. |
| `nnqc/train.py` | Autoencoder and diffusion training loops; contains `EMA`, corruption sampling, and DDP logic. |
| `nnqc/evaluate.py` | DDIM sampling and reconstruction visualization panels. |
| `nnqc/infer.py` | `check()` and `QCResult`; preprocessing, inverse transforms, and metric aggregation. |
| `nnqc/hub.py` | Zenodo weight downloader: downloads the shared zip archive once, extracts `weight_<task>/` (stdlib `urllib` + `zipfile`, no token needed). |
| `nnqc/metrics.py` | Pluggable `Metric` base class and built-ins (`dice`, `iou`, medpy adapters). |
| `nnqc/xa.py` | `CLIPCrossAttentionGrid`: UniMed-CLIP wrapper that produces cross-attention context. **CUDA-only.** |
| `nnqc/corruptions.py` | `corrupt_ohe_masks_v2`: signed-distance-transform based realistic corruptions. |
| `nnqc/utils.py` | Dataloaders, transforms, helpers, and some unused legacy code. |
| `nnqc/refine.py` | Post-processing: hole filling, largest component, z smoothing. |
| `nnqc/visualize.py` | TensorBoard image helpers. |
| `nnqc/mcp_server.py` | MCP stdio server (`list_tasks`, `check_mask`, `explain_qc_score`). |
| `nnqc/maisi_vae.py` | MAISI VAE support utilities. |

### Important architectural notes

- **Slice-ratio embedding** (`embed`): `Linear(1, 32) → GELU → Linear(32, cross_attention_dim)`. It is constructed identically in `train.py`, `evaluate.py`, and `infer.py`; keep all three in sync if you change it.
- **Conditioning context** is built as `[scan_token, position_token]` in `xa.py`. Do **not** route the position embedding through the old `CrossAttentionGrid` path with `column_softmax` over a length-1 axis — that made the output independent of slice position.
- **Corruption model:** training uses `corruptions.corrupt_ohe_masks_v2`, which perturbs the signed distance transform. The legacy `utils.corrupt_ohe_masks` (rectangular holes) is unused.
- **Intensity scaling:** the historical default does **not** clip intensities. Setting `clip_intensity: true` changes the input distribution; do not flip it under an existing checkpoint.
- **DDIM steps:** more is not better. Per-anatomy measured optima are cardiac **2**, prostate/liver **4–5**. `check(num_steps=None)` reads `inference.num_steps` from the config and falls back to 5.
- **Autoencoder in fp32:** the AE is trained in fp32 because its `norm_eps=1e-6` is subnormal in fp16 and produces NaN under autocast. `train._ae_fp32` enforces this. Do not move the AE back into the autocast region.
- **Resume semantics:** `--resume --start-epoch N` loads `_last.pt` checkpoints, fast-forwards the LR scheduler, and preserves `diffusion_best_val.txt` so a worse resumed validation cannot overwrite the best checkpoint.
- **Candidate-state gate (`xa_mask_gate`):** off by default. Enabling it creates a fresh regime; no trained gated model exists yet.

## Development conventions

- Keep heavy imports (torch, monai) out of global scope in `__init__.py` and `cli.py`. The `_LAZY` table in `__init__.py` and the branch imports in `cli.py:main` exist so `nnqc list-tasks` and `nnqc --version` stay fast.
- User-visible log lines are prefixed `[nnqc]`.
- TensorBoard is the only metric sink; rank 0 writes to `<tfevent_path>/{autoencoder,diffusion,eval}`.
- Batch size semantics: `batch_size` is the number of *slices* per optimizer step. The dataloader iterates volumes with `batch_size=1` and uses `RandSpatialCropSamplesd(num_samples=batch_size)` to sample slices from each volume.
- Data splits: `prepare_general_dataloader` does an 80/20 train/val split at seed 42; `prepare_msd_dataloader` uses MONAI `DecathlonDataset`.
- When adding a geometric transform to the preprocessing chain, add it to the inverse loop in `infer.py` as well.
- Do not commit trained weights, datasets, logs, or outputs. These are all gitignored.

## Testing strategy

- Tests live in `tests/` and use `pytest`.
- The `slow` marker is defined in `pytest.ini` for CPU-heavy signed-distance-transform tests. Deselect with `-m "not slow"`.
- Key regression areas have dedicated test files:
  - `tests/test_amp_nan.py` — autoencoder NaN guard under autocast.
  - `tests/test_conditioning.py` — slice-position embedding reaches the model.
  - `tests/test_benchmark_metrics.py` — aggregate metrics do not hide bad behavior.
  - `tests/test_refine.py` — hole filling preserves nested classes.
  - `tests/test_sampling.py` — DDIM sampling / ensembling and per-task step counts.
  - `tests/test_maisi_vae.py` — MAISI VAE path.

Run the fast suite before committing:

```bash
pytest tests/ -q -m "not slow"
```

## Deployment and operations

### CLI

```bash
nnqc list-tasks
nnqc train-autoencoder --task prostate --epochs 500 --lr 5e-5 --device 0
nnqc train-diffusion --task prostate --epochs 4000 --lr 2.5e-5 --scheduler cosine --device 0
nnqc evaluate --task prostate --num-volumes 3 --num-steps 5 --device 0
nnqc check --task prostate --image scan.nii.gz --mask candidate_mask.nii.gz --save recon.nii.gz
nnqc download prostate
```

### Slurm

Slurm scripts live in `slurm/`:

```bash
sbatch slurm/smoke_nnqc.sh cardiac        # ~10 min end-to-end check on 4×H100
sbatch slurm/probe_ae.sh                  # stage-1 ceiling: AE round-trip Dice vs GT
sbatch slurm/benchmark_nnqc.sh prostate   # recon Dice + QC calibration
bash slurm/launch_campaign.sh             # AE -> diffusion -> resume legs, all tasks
```

`slurm/env_common.sh` is sourced by every batch job. It activates the venv, loads `arch/h100`, sets offline Hub variables, and prepends torch's bundled CUDA libraries to `LD_LIBRARY_PATH`.

### MCP server

`nnqc/mcp_server.py` exposes QC over stdio:

```bash
python -m nnqc.mcp_server
```

Tools: `list_tasks`, `check_mask`, `explain_qc_score`. stdout is the JSON-RPC transport, so `nnqc` prints must be redirected to stderr (`NNQC_MCP_STRICT_STDOUT=1` can make protocol violations raise).

## Security considerations

- **Tokens:** pretrained weights are public on Zenodo and need no token. Do not hardcode tokens in source files or commit them.
- **Compute nodes are offline:** Slurm jobs run with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`. Any required assets must be cached on a login node before submission.
- **Do not commit sensitive data:** `.gitignore` excludes trained weights (`*.pt`, `*.pth`, `*.safetensors`), datasets (`*.nii.gz`, `dataset/`, `data/`), logs, and local environment notes.
- **Weight files can be large (>1 GB per task).** They are distributed separately via Zenodo, not via git.

## Common pitfalls

- `xa.py` is **CUDA-only** and calls `.cuda()` unconditionally; there is no CPU inference path.
- The UniMed-CLIP backbone checkpoint must be pre-sanitized with `scripts/prepare_backbone.py` because the published checkpoint contains optimizer/scaler state that `torch>=2.6` rejects under `weights_only=True`.
- `scale_factor` is written to `<model_dir>/scale_factor.txt` by training and read back by inference; pre-existing checkpoints fall back to estimating it from the mask being judged.
- `find_paired_files` requires an exact identifier match (`MIN_PAIR_SCORE = 10`). Stage data as `<case>_img` / `<case>_gt` so the intended label wins.
- Autoencoder LR is multiplied by `world_size` in DDP; diffusion LR is not.
- Changing `clip_intensity`, `diffusion_train.best_metric`, or `val_severities` changes the training regime and resets the best checkpoint tracking.

## Useful references

- `README.md` — high-level overview and quickstarts.
- `CLAUDE.md` — detailed operational notes for Claude Code, including measured findings and sharp edges.
- `tutorials/TUTORIAL.md` — end-to-end walkthrough.
- `results/FINDINGS.md` — measured results and known limitations (read before re-litigating model behavior).
- `pyproject.toml` — dependencies, build config, and ruff rules.
