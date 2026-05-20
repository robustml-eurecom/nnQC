#!/usr/bin/env python3
"""nnQC v2 calibration: per-subject Dice(GT, corruption) vs Dice(pgt, corruption).

For each subject (volume) in the val/test loader:
 - generate `--num-candidates` random corruptions with random intensity
 - keep the `--num-corruptions` whose Dice(GT, corr) is closest to the
    `--target-dices` (default 0.10, 0.31, 0.52, 0.73, 0.95)
 - run the diffusion model (DDIM `--num-steps`) on each corrupted volume
 - compute Dice(pgt, corr) - this is the nnQC "predicted quality" signal
 - record (dice_true, dice_pred); aggregate MAE

Outputs:
 - <out-dir>/results.csv - one row per (subject, target) pair
 - <out-dir>/scatter.png - Dice(pgt, corr) vs Dice(GT, corr)
 - <out-dir>/per_subject_<i>.png - 5 corruption panels for inspection
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import AsDiscrete as OHE
from monai.utils import first, set_determinism
from torch.amp import autocast

from nnqc.corruptions import DEFAULT_CFG, corrupt_ohe_masks_v2
from nnqc.utils import compute_spacing, define_instance, prepare_general_dataloader, prepare_msd_dataloader
from nnqc.xa import CLIPCrossAttentionGrid


def parse_args():
    p = argparse.ArgumentParser(description="nnQC v2 Dice calibration / MAE")
    p.add_argument("-c", "--config-file", default="./config/config_prostate_v2.json")
    p.add_argument("-e", "--environment-file", default="./config/env_prostate_v2.json")
    p.add_argument("--checkpoint-suffix", choices=["last", "best"], default="last")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="which loader to evaluate on (val is usually larger)")
    p.add_argument("--num-subjects", type=int, default=0,
                   help="0 = all subjects in the chosen split")
    p.add_argument("--num-corruptions", type=int, default=5)
    p.add_argument("--num-candidates", type=int, default=30,
                   help="random corruption draws per subject to bin into target Dice levels")
    p.add_argument("--target-dices", type=float, nargs="+",
                   default=[0.10, 0.31, 0.52, 0.73, 0.95])
    p.add_argument("--num-steps", type=int, default=5, help="DDIM sampling steps")
    p.add_argument("--inference-batch", type=int, default=16,
                   help="slices per forward pass during DDIM")
    p.add_argument("--out-dir", type=str, default=None,
                   help="default: <output_dir>/qc_calibration")
    p.add_argument("--merge-foreground-classes", action="store_true",
                   help="for multi-class models: collapse all foreground "
                        "channels into a single binary mask before Dice")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def random_intensity_config(rng):
    """Random tweaks on DEFAULT_CFG to span a wide Dice range across draws."""
    intensity = rng.random()
    cfg = {}
    cfg["elastic_prob"] = 0.5
    cfg["elastic_sigma"] = (3.0, 5.0 + 10.0 * intensity)
    cfg["elastic_magnitude"] = (1.0, 2.0 + 8.0 * intensity)
    cfg["boundary_prob"] = 0.4
    cfg["boundary_noise_std"] = (0.5, 1.0 + 4.0 * intensity)
    cfg["boundary_smooth_sigma"] = (4.0, 8.0 + 8.0 * intensity)
    cfg["erosion_prob"] = 0.3 + 0.3 * intensity
    cfg["erosion_fraction"] = (0.02, 0.10 + 0.40 * intensity)
    cfg["dilation_prob"] = 0.3 + 0.3 * intensity
    cfg["dilation_voxels"] = (0.5, 1.5 + 6.0 * intensity)
    cfg["blob_fp_prob"] = 0.2 + 0.5 * intensity
    cfg["blob_num"] = (1, 1 + int(3 * intensity))
    cfg["blob_size"] = (0.05, 0.10 + 0.40 * intensity)
    cfg["split_prob"] = 0.1 + 0.2 * intensity
    cfg["max_operations"] = 1 + int(3 * intensity)
    return cfg


def _binary_dice(a_b, b_b, eps=1e-6):
    inter = (a_b * b_b).sum()
    denom = a_b.sum() + b_b.sum()
    return (2 * inter / (denom + eps)).item()


def volume_dice(a, b, num_classes, merge_foreground=False, eps=1e-6):
    """Dice between two tensors.
    a, b: [N, 1, H, W] (binary) or [N, C, H, W] (multi-class one-hot).
    If merge_foreground=True (or num_classes==1), return a single binary Dice
    on the union of all foreground channels."""
    if num_classes == 1:
        return _binary_dice((a > 0.5).float(), (b > 0.5).float(), eps)
    if merge_foreground:
        # foreground = any non-background channel active
        a_b = (a[:, 1:].sum(dim=1) > 0.5).float()
        b_b = (b[:, 1:].sum(dim=1) > 0.5).float()
        return _binary_dice(a_b, b_b, eps)
    # multi-class average: skip channel 0 (background)
    dices = []
    for ch in range(1, num_classes):
        a_b = (a[:, ch] > 0.5).float()
        b_b = (b[:, ch] > 0.5).float()
        if (a_b.sum() + b_b.sum()) < 1e-3:
            continue
        dices.append(_binary_dice(a_b, b_b, eps))
    return float(np.mean(dices)) if dices else float("nan")


def ddim_sample(unet, autoencoder, scheduler, xa, embed,
                images_ohe, scans, slice_ratios, num_classes, num_steps,
                scale_factor, batch_size, device, seed):
    """Run DDIM denoising for an entire volume of slices, returning pgt logits/onehot."""
    N = images_ohe.shape[0]
    out = torch.zeros_like(images_ohe)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        img_b = images_ohe[s:e]
        scan_b = scans[s:e]
        r_b = slice_ratios[s:e].unsqueeze(1)

        # corrupted-mask input is built outside this fn; we expect images_ohe
        # already to be the *corrupted* mask in OHE form
        if num_classes > 1:
            corr_in = img_b.argmax(1, keepdim=True).float() / num_classes
        else:
            corr_in = img_b.float()

        slice_emb = embed(r_b).float()
        c, _, _ = xa(scan_b, ext_features=slice_emb)
        c = c.float().unsqueeze(1)

        # latent shape for noise (encode a zeros prior so we know the shape)
        zeros_for_shape = torch.zeros_like(img_b) if num_classes > 1 else img_b
        z_shape = autoencoder.encode_stage_2_inputs(zeros_for_shape).shape
        mask_resized = F.interpolate(corr_in, size=z_shape[2:], mode="nearest")

        gen = torch.Generator(device=device).manual_seed(int(seed) + s)
        z_t = torch.randn(z_shape, generator=gen, device=device, dtype=img_b.dtype)
        scheduler.set_timesteps(num_steps)
        for t in scheduler.timesteps:
            t_b = torch.full((z_t.shape[0],), int(t.item()), device=device, dtype=torch.long)
            eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=c)
            z_t = scheduler.step(eps, t, z_t)[0]
        decoded = autoencoder.decode_stage_2_outputs(z_t / scale_factor)

        if num_classes > 1:
            # convert logits -> one-hot prediction
            pred = decoded.argmax(1)
            out[s:e] = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
        else:
            out[s:e, 0] = (decoded[:, 0].sigmoid() > 0.5).float()
    return out


def main():
    args = parse_args()
    env = json.load(open(args.environment_file))
    cfg = json.load(open(args.config_file))
    for k, v in env.items():
        setattr(args, k, v)
    for k, v in cfg.items():
        setattr(args, k, v)

    set_determinism(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    print(f"Using {device} ({torch.cuda.get_device_name(0)})")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_dir) / "qc_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {out_dir}")

    # --- dataloader ---
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    spacing = compute_spacing("dataset", args, save=False)
    if getattr(args, "is_msd", False):
        loaders = prepare_msd_dataloader(
            args, args.autoencoder_train["batch_size"], args.autoencoder_train["patch_size"], spacing,
            sample_axis=args.sample_axis, randcrop=True, world_size=1, cache=0.0,
            download=False, size_divisible=size_divisible,
        )
    else:
        loaders = prepare_general_dataloader(
            args, args.image_pattern, args.label_pattern,
            args.autoencoder_train["batch_size"], args.autoencoder_train["patch_size"], spacing,
            sample_axis=args.sample_axis, randcrop=True, world_size=1, cache=0.0,
            size_divisible=size_divisible,
        )
    # choose split
    val_loader = loaders[1]
    test_loader = loaders[2] if len(loaders) >= 3 else None
    chosen_loader = test_loader if (args.split == "test" and test_loader is not None) else val_loader
    if args.split == "test" and test_loader is None:
        print("[warn] no test loader - falling back to val")

    ohe = OHE(to_onehot=args.num_classes, dim=1)

    # --- models ---
    print("Loading models...")
    autoencoder = define_instance(args, "autoencoder_def").to(device).eval()
    autoencoder.load_state_dict(torch.load(
        os.path.join(args.model_dir, "autoencoder.pt"), map_location=device, weights_only=True))

    unet = define_instance(args, "diffusion_def").to(device).eval()
    ckpt_name = "diffusion_unet.pt" if args.checkpoint_suffix == "best" else "diffusion_unet_last.pt"
    unet.load_state_dict(torch.load(
        os.path.join(args.model_dir, ckpt_name), map_location=device, weights_only=True))
    print(f"  loaded UNet from {ckpt_name}")

    xa = CLIPCrossAttentionGrid(
        output_dim=args.diffusion_def["cross_attention_dim"], grid_reduction="column_softmax",
    ).to(device).eval()
    xa_name = "xa.pt" if args.checkpoint_suffix == "best" else "xa_last.pt"
    xa.load_state_dict(torch.load(
        os.path.join(args.model_dir, xa_name), map_location=device, weights_only=True))

    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, args.diffusion_def["cross_attention_dim"]),
    ).to(device).eval()
    embed_name = "embed.pt" if args.checkpoint_suffix == "best" else "embed_last.pt"
    embed.load_state_dict(torch.load(
        os.path.join(args.model_dir, embed_name), map_location=device, weights_only=True))

    # scale factor (recompute as in training)
    with torch.no_grad(), autocast("cuda", enabled=True):
        check = first(chosen_loader)["label"].to(device)
        if args.num_classes > 1:
            check = ohe(check)
        z = autoencoder.encode_stage_2_inputs(check)
        scale_factor = 1.0 / torch.std(z)
    print(f"  scale_factor = {scale_factor.item():.4f}")

    scheduler = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=args.NoiseScheduler["clip_sample"],
    )

    target_dices = list(args.target_dices)[: args.num_corruptions]
    rows = []   # (subject_idx, target, dice_true, dice_pred, abs_err)

    with torch.no_grad(), autocast("cuda", enabled=True):
        for vol_idx, batch in enumerate(chosen_loader):
            if args.num_subjects and vol_idx >= args.num_subjects:
                break
            images = batch["label"].to(device)            # [N, 1, H, W]
            if args.num_classes > 1:
                images_ohe = ohe(images)
            else:
                images_ohe = images
            scans = batch["image"].to(device).float()
            slice_ratios = batch["slice_label"].float().to(device)
            n_slices = images_ohe.shape[0]
            if n_slices < 1:
                continue

            # --- 1) generate candidate corruptions, record their Dice ---
            candidates = []
            for k in range(args.num_candidates):
                cfg_k = random_intensity_config(rng)
                corr_ohe = corrupt_ohe_masks_v2(images_ohe, corruption_prob=1.0, config=cfg_k)
                d_true = volume_dice(images_ohe, corr_ohe, args.num_classes,
                                    merge_foreground=args.merge_foreground_classes)
                if np.isnan(d_true):
                    continue
                candidates.append((d_true, corr_ohe.detach()))
            if not candidates:
                print(f"  vol {vol_idx}: no valid corruptions")
                continue

            # --- 2) pick the candidate closest to each target Dice ---
            picks = []
            taken = set()
            for tgt in target_dices:
                best = None
                for ci, (d, c) in enumerate(candidates):
                    if ci in taken:
                        continue
                    diff = abs(d - tgt)
                    if best is None or diff < best[0]:
                        best = (diff, ci, d, c)
                if best is not None:
                    taken.add(best[1])
                    picks.append((tgt, best[2], best[3]))

            # --- 3) run model on each picked corruption ---
            panel_rows = []
            for (tgt, d_true, corr_ohe) in picks:
                pgt = ddim_sample(
                    unet, autoencoder, scheduler, xa, embed,
                    corr_ohe, scans, slice_ratios, args.num_classes,
                    args.num_steps, scale_factor, args.inference_batch, device,
                    seed=args.seed + vol_idx,
                )
                d_pred = volume_dice(pgt, corr_ohe, args.num_classes,
                                     merge_foreground=args.merge_foreground_classes)
                abs_err = abs(d_true - d_pred)
                rows.append((vol_idx, tgt, d_true, d_pred, abs_err))
                panel_rows.append((tgt, d_true, d_pred, corr_ohe, pgt))
                print(f"  vol {vol_idx} | tgt={tgt:.2f}  Dice(GT,corr)={d_true:.3f}  Dice(pgt,corr)={d_pred:.3f}  |Δ|={abs_err:.3f}")

            # --- 4) per-subject panel ---
            mid = n_slices // 2
            n_panels = len(panel_rows)
            fig, axes = plt.subplots(3, n_panels, figsize=(2.4 * n_panels, 7.2))
            if n_panels == 1:
                axes = axes[:, None]
            for col, (tgt, d_true, d_pred, corr, pgt) in enumerate(panel_rows):
                scan_img = scans[mid, 0].cpu().numpy()
                if args.num_classes > 1 and not args.merge_foreground_classes:
                    gt = images_ohe[mid].argmax(0).cpu().numpy()
                    cr = corr[mid].argmax(0).cpu().numpy()
                    pr = pgt[mid].argmax(0).cpu().numpy()
                    kw = dict(cmap="tab10", vmin=0, vmax=args.num_classes - 1)
                elif args.num_classes > 1:
                    gt = (images_ohe[mid, 1:].sum(0) > 0.5).cpu().numpy().astype(np.uint8)
                    cr = (corr[mid, 1:].sum(0) > 0.5).cpu().numpy().astype(np.uint8)
                    pr = (pgt[mid, 1:].sum(0) > 0.5).cpu().numpy().astype(np.uint8)
                    kw = dict(cmap="gray", vmin=0, vmax=1)
                else:
                    gt = images_ohe[mid, 0].cpu().numpy()
                    cr = corr[mid, 0].cpu().numpy()
                    pr = pgt[mid, 0].cpu().numpy()
                    kw = dict(cmap="gray", vmin=0, vmax=1)
                axes[0, col].imshow(scan_img, cmap="gray")
                axes[0, col].imshow(gt, alpha=0.4, **kw)
                axes[1, col].imshow(cr, **kw)
                axes[2, col].imshow(pr, **kw)
                axes[0, col].set_title(f"tgt={tgt:.2f}\nD(GT,c)={d_true:.2f}")
                axes[2, col].set_title(f"D(pgt,c)={d_pred:.2f}")
                for ax in axes[:, col]:
                    ax.set_xticks([]); ax.set_yticks([])
            axes[0, 0].set_ylabel("scan+GT")
            axes[1, 0].set_ylabel("corruption")
            axes[2, 0].set_ylabel("pgt (model)")
            fig.suptitle(f"vol {vol_idx} - DDIM {args.num_steps} steps", y=1.0)
            fig.tight_layout()
            fig.savefig(out_dir / f"per_subject_{vol_idx:02d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    # --- write CSV + scatter + MAE ---
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "target_dice", "dice_gt_corr", "dice_pgt_corr", "abs_err"])
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    if rows:
        d_true_a = np.array([r[2] for r in rows])
        d_pred_a = np.array([r[3] for r in rows])
        err = np.array([r[4] for r in rows])
        mae = float(err.mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        corr = float(np.corrcoef(d_true_a, d_pred_a)[0, 1]) if len(rows) > 1 else float("nan")
        print(f"\n=== {len(rows)} (subject, target) pairs ===")
        print(f"  MAE  |Dice(GT,corr) - Dice(pgt,corr)| = {mae:.4f}")
        print(f"  RMSE                                  = {rmse:.4f}")
        print(f"  Pearson r                              = {corr:.4f}")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(d_true_a, d_pred_a, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Dice(GT, corruption)")
        ax.set_ylabel("Dice(pgt, corruption)")
        ax.set_title(f"nnQC calibration - MAE={mae:.3f}, r={corr:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=130)
        plt.close(fig)
        print(f"  scatter saved to {out_dir / 'scatter.png'}")


if __name__ == "__main__":
    main()
