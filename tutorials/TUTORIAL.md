# nnQC tutorial - end-to-end on MSD Prostate and MSD Spleen

This tutorial walks through:

1. preparing the dataset,
2. training the autoencoder,
3. training the diffusion UNet,
4. visualizing reconstructions,
5. computing the calibration MAE between `Dice(GT, corruption)` and the
   nnQC predicted Dice `Dice(pgt, corruption)`.

All commands assume the working directory is the repo root.

---

## 0. Setup

```bash
git clone https://github.com/robustml-eurecom/nnQC.git
cd nnQC
uv venv && source .venv/bin/activate
uv pip install -e .

# Download the UniMedCLIP backbone (referenced by xa.py).
# Place it at: trained_weights/unimed_clip_vit_b16.pt
```

---

## 1. Data layout

### Prostate (general loader)

`config/env_prostate.json` expects a directory tree under
`./dataset/<task>/` with paired files matched by glob:

```
dataset/Task05Prostate_flat/
├── prostate_00_t2.nii.gz       # image_pattern: *_t2.nii.gz
├── prostate_00_gt.nii.gz       # label_pattern: *_gt.nii.gz
├── prostate_01_t2.nii.gz
├── prostate_01_gt.nii.gz
└── …
```

The split is 80% train / 20% val with a fixed seed (42).

### Spleen (MSD loader)

`config/env_spleen.json` uses MONAI's MSD layout:

```
misc_dataset/Task09_Spleen/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

Set `is_msd: true` in the env file and MONAI will discover the volumes
automatically.

---

## 2. Stage 1 - Autoencoder

Train the AutoencoderKL on one-hot segmentation masks. With
`num_classes = 3` (prostate) the AE has 3 in/out channels; with
`num_classes = 1` (spleen) it has a single channel.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_autoencoder.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    -g 1
```

Look at `<tfevent_path>/autoencoder/` in TensorBoard for the recon Dice
curve - it should plateau quickly (around 0.01 Dice loss).

The script writes:

- `<model_dir>/autoencoder.pt` - best by val
- `<model_dir>/autoencoder_last.pt` - most recent
- `<model_dir>/discriminator.pt` (+ `_last`) - adversarial discriminator
  used during AE training

You can stop once the val curve plateaus.

---

## 3. Stage 2 - Diffusion UNet

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_diffusion.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    -g 1
```

What this script does each epoch:

1. Samples a batch of *clean* masks, encodes them to the latent space.
2. Generates a *corrupted* mask via the v2 morphologically-realistic
   corruption suite (see `nnqc/corruptions.py`).
3. Computes the cross-attention context: UniMedCLIP image features of the
   scan, fused with a slice-ratio MLP embedding (so the model knows
   apex vs base).
4. Adds Gaussian noise to the latent at a random timestep.
5. Trains the UNet (in_channels = latent_dim + 1) to predict the noise
   from `concat(noisy_latent, corrupted_mask_resized)`.
6. From epoch `warmup_dice_epochs` onward, adds a Dice term on the
   decoded x0 estimate (`lambda_recon * GeneralizedDiceLoss(seg_pred, GT)`).

Key TensorBoard scalars:

- `train_diffusion_loss_iter`, `..._ema` - instantaneous and EMA-smoothed train loss
- `val_diffusion_loss_raw` vs `val_diffusion_loss_ema` - raw model vs EMA weights
- `val_diffusion_loss_2` - reconstruction Dice (after warm-up)

The EMA of the UNet (`decay=0.999`) is what gets saved as both `.pt` (best
by val loss) and `_last.pt`. EMA weights are what you should sample from
at inference.

---

## 4. Visualize reconstructions

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_visualize.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    --num-volumes 3 --num-steps 5
```

For each val volume it picks the apex / mid / base slice (by
`slice_ratio`), corrupts the GT mask, and runs DDIM denoising. Output:

- `<output_dir>/eval_step_<step>/vol{00,01,02}.png` - 3×4 grid:
  `scan | corrupted | GT | sample`
- TensorBoard figures under `<tfevent_path>/eval/`

If you want a sweep over different denoising lengths in one shot:

```bash
python scripts/sweep_ddim_steps.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    --steps-list 5 10 20 30 40 50
```

Empirically, 5 DDIM steps gave the cleanest samples on our prostate
checkpoint; longer chains added subtle artifacts. Re-check on your own
checkpoint.

---

## 5. QC calibration

The whole point of nnQC: the discrepancy between an input mask and the
model's reconstruction should track the true error against the GT.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_qc_calibration.py \
    -c configs/prostate/config.json \
    -e configs/prostate/env.json \
    --split val \
    --num-corruptions 5 \
    --num-candidates 25 \
    --num-steps 5
```

Per subject the script:

1. Samples 25 candidate corruptions with randomized intensity.
2. Picks 5 whose `Dice(GT, corruption)` is closest to the target Dice
   levels (default `[0.10, 0.31, 0.52, 0.73, 0.95]`).
3. Runs the model (DDIM 5 steps) and computes `Dice(pgt, corruption)`.

Outputs in `<output_dir>/qc_calibration/`:

| File | Content |
|------|---------|
| `results.csv` | subject, target, Dice(GT,corr), Dice(pgt,corr), abs_err |
| `scatter.png` | scatter of Dice(pgt,corr) vs Dice(GT,corr) with the identity line |
| `per_subject_*.png` | 3-row panel per subject (scan+GT, corruption, pgt) for each corruption level |
| stdout | overall MAE, RMSE, Pearson r |

For multi-class models you can also collapse foreground channels into a
single binary mask before computing Dice:

```bash
python scripts/eval_qc_calibration.py … --merge-foreground-classes \
    --out-dir output/prostate_v2/qc_calibration_binary
```

In our reference run the prostate v2 model was poorly calibrated at the
per-class level (MAE 0.18) but **well calibrated as a binary foreground
QC signal** (MAE 0.04, Pearson r 0.78).

### What you should see on a healthy model

- **Scatter plot** clustered near the identity line `y = x`.
- **MAE < 0.1** on the val split.
- **Pearson r > 0.7** between true and predicted Dice.

If a single subject's points sit well off the line (as vol 0 did in our
prostate calibration), inspect its `per_subject_*.png`: usually the
model has placed the prostate in the wrong anatomical location because
the scan / slice-ratio conditioning was misled.

---

## 6. Where to go from here

- **Reproduce on your own task**: copy `configs/spleen/` to
  `configs/my_task/`, edit `env.json` to point at your data, edit
  `num_classes` in `config.json`. The training scripts auto-branch
  between `prepare_general_dataloader` (paired files) and
  `prepare_msd_dataloader` (MSD format) on `is_msd`.
- **Tighten the calibration**: the Dice term (`lambda_recon`) is the
  main lever for forcing pgt to match GT spatially. Try `0.3` if your
  scatter has too much vertical spread at high target Dice.
- **Cheaper sampling**: 5 DDIM steps are sufficient in our setting. If
  you reduce further (e.g., 2-3 steps), validate on the calibration
  scatter before deploying.
