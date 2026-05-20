#!/usr/bin/env python3
"""Periodic evaluation + visualization for the nnQC v2 prostate diffusion model.

Loads the latest checkpoint from <model_dir>, runs DDIM sampling on a few
validation volumes, and renders apex/mid/base reconstruction panels
(scan | corrupted input | GT | sampled reconstruction).

Designed to run alongside training on a *different* GPU; intended to be
re-invoked once every 10 epochs by `watch_eval.sh`.

Outputs:
 - PNGs to  <output_dir>/eval_step_<step>/
 - TensorBoard figures under  <tfevent_path>/eval/

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python eval_visualize.py --step 30
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import AsDiscrete as OHE
from monai.utils import first, set_determinism
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from nnqc.corruptions import corrupt_ohe_masks_v2
from nnqc.utils import compute_spacing, define_instance, prepare_general_dataloader, prepare_msd_dataloader
from nnqc.xa import CLIPCrossAttentionGrid


def parse_args():
    p = argparse.ArgumentParser(description="nnQC v2 eval + viz")
    p.add_argument("-c", "--config-file", default="./config/config_prostate_v2.json")
    p.add_argument("-e", "--environment-file", default="./config/env_prostate_v2.json")
    p.add_argument("--checkpoint-suffix", choices=["last", "best"], default="last",
                   help="last = diffusion_unet_last.pt (most recent), best = diffusion_unet.pt")
    p.add_argument("--num-volumes", type=int, default=3, help="how many val volumes to render")
    p.add_argument("--num-steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--step", type=int, default=0, help="TensorBoard step (typically the train epoch)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    env = json.load(open(args.environment_file))
    cfg = json.load(open(args.config_file))
    for k, v in env.items():
        setattr(args, k, v)
    for k, v in cfg.items():
        setattr(args, k, v)

    set_determinism(args.seed)
    device = torch.device("cuda")
    print(f"Using {device} ({torch.cuda.get_device_name(0)})")

    out_dir = Path(args.output_dir) / f"eval_step_{args.step:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tfevent_path) / "eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # --- val dataloader (no caching: each invocation only reads a few volumes) ---
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    spacing = compute_spacing("dataset", args, save=False)
    if getattr(args, "is_msd", False):
        _, val_loader = prepare_msd_dataloader(
            args, args.autoencoder_train["batch_size"], args.autoencoder_train["patch_size"], spacing,
            sample_axis=args.sample_axis, randcrop=True, world_size=1, cache=0.0,
            download=False, size_divisible=size_divisible,
        )
    else:
        _, val_loader, _ = prepare_general_dataloader(
            args, args.image_pattern, args.label_pattern,
            args.autoencoder_train["batch_size"], args.autoencoder_train["patch_size"], spacing,
            sample_axis=args.sample_axis, randcrop=True, world_size=1, cache=0.0,
            size_divisible=size_divisible,
        )

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

    # --- scale factor (recompute as in training) ---
    with torch.no_grad(), autocast("cuda", enabled=True):
        check = first(val_loader)["label"].to(device)
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

    # --- visualize a few val volumes ---
    figures_logged = 0
    for vol_idx, batch in enumerate(val_loader):
        if vol_idx >= args.num_volumes:
            break
        images = batch["label"].to(device)            # [N, 1, H, W]
        if args.num_classes > 1:
            images = ohe(images)                      # [N, C, H, W]
        scans = batch["image"].to(device).float()     # [N, 1, H, W]
        slice_ratios = batch["slice_label"].float().to(device)  # [N]
        n = images.shape[0]
        if n < 3:
            print(f"  vol {vol_idx}: only {n} slices, skipped")
            continue

        sorted_idx = torch.argsort(slice_ratios).tolist()
        picks = [
            ("apex", sorted_idx[0]),
            ("mid",  sorted_idx[len(sorted_idx) // 2]),
            ("base", sorted_idx[-1]),
        ]
        sub_idx = [i for _, i in picks]
        images_s = images[sub_idx]
        scans_s = scans[sub_idx]
        ratios_s = slice_ratios[sub_idx].unsqueeze(1)

        with torch.no_grad(), autocast("cuda", enabled=True):
            corr_mask = corrupt_ohe_masks_v2(images_s, corruption_prob=1.0)
            if args.num_classes > 1:
                corr_in = corr_mask.argmax(1, keepdim=True).float() / args.num_classes
            else:
                corr_in = corr_mask.float()

            slice_embeddings = embed(ratios_s).float()
            c, _, _ = xa(scans_s, ext_features=slice_embeddings)
            c = c.float().unsqueeze(1)

            z_enc = autoencoder.encode_stage_2_inputs(images_s) * scale_factor
            mask_resized = F.interpolate(corr_in, size=z_enc.shape[2:], mode="nearest")
            z_t = torch.randn(z_enc.shape, device=device, dtype=z_enc.dtype)

            scheduler.set_timesteps(args.num_steps)
            for t in scheduler.timesteps:
                t_b = torch.full((z_t.shape[0],), int(t.item()), device=device, dtype=torch.long)
                eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=c)
                # MONAI's DDIMScheduler.step returns (prev_sample, pred_original_sample)
                z_t = scheduler.step(eps, t, z_t)[0]

            sampled = autoencoder.decode_stage_2_outputs(z_t / scale_factor)

        # --- figure: rows = apex/mid/base, cols = scan | corrupted | GT | sample ---
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for row, (name, idx) in enumerate(picks):
            scan_img = scans_s[row, 0].cpu().numpy()
            if args.num_classes > 1:
                corr_lbl = corr_mask[row].argmax(0).cpu().numpy()
                gt_lbl = images_s[row].argmax(0).cpu().numpy()
                samp_lbl = sampled[row].argmax(0).cpu().numpy()
                kw = dict(cmap="tab10", vmin=0, vmax=args.num_classes - 1)
            else:
                corr_lbl = corr_mask[row, 0].cpu().numpy()
                gt_lbl = images_s[row, 0].cpu().numpy()
                samp_lbl = (sampled[row, 0].sigmoid() > 0.5).cpu().numpy()
                kw = dict(cmap="gray", vmin=0, vmax=1)

            axes[row, 0].imshow(scan_img, cmap="gray")
            axes[row, 1].imshow(corr_lbl, **kw)
            axes[row, 2].imshow(gt_lbl, **kw)
            axes[row, 3].imshow(samp_lbl, **kw)
            axes[row, 0].set_ylabel(f"{name}\nr={slice_ratios[idx].item():.2f}", fontsize=10)
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].set_title("scan")
        axes[0, 1].set_title("corrupted")
        axes[0, 2].set_title("GT")
        axes[0, 3].set_title(f"sample (DDIM {args.num_steps})")
        fig.suptitle(f"nnQC v2 - volume {vol_idx} - step {args.step}", y=1.0)
        fig.tight_layout()

        png_path = out_dir / f"vol{vol_idx:02d}.png"
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        writer.add_figure(f"eval/vol{vol_idx:02d}", fig, args.step)
        plt.close(fig)
        figures_logged += 1
        print(f"  vol {vol_idx}: saved {png_path.name}")

    writer.flush()
    writer.close()
    print(f"Done. {figures_logged} figures in {out_dir}  +  TensorBoard at {tb_dir}")


if __name__ == "__main__":
    main()
