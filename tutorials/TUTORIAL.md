# nnQC tutorial - end-to-end on MSD Prostate and MSD Spleen

This tutorial walks through:

1. preparing the dataset,
2. training the autoencoder,
3. training the diffusion UNet,
4. visualizing reconstructions.

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

- `<output_dir>/eval_step_<step>/vol{00,01,02}.png` - 3x4 grid:
  `scan | corrupted | GT | sample`
- TensorBoard figures under `<tfevent_path>/eval/`

Empirically, 5 DDIM steps gave the cleanest samples on our reference
checkpoints; longer chains added subtle artifacts. Re-check on your own
checkpoint.

---

## 5. Where to go from here

- **Reproduce on your own task**: copy `configs/spleen/` to
  `configs/my_task/`, edit `env.json` to point at your data, edit
  `num_classes` in `config.json`. The training scripts auto-branch
  between `prepare_general_dataloader` (paired files) and
  `prepare_msd_dataloader` (MSD format) on `is_msd`.
- **Cheaper sampling**: 5 DDIM steps are sufficient in our setting. If
  you reduce further (e.g., 2-3 steps), validate visual quality on the
  apex / mid / base panels before deploying.
