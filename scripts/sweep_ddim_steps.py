#!/usr/bin/env python3
"""One-shot DDIM step-count sweep for nnQC v2 - no TensorBoard logging.

For each of N validation volumes, picks apex/mid/base slices and renders a
side-by-side grid of samples produced with [5, 10, 20, 30, 40, 50] DDIM steps
from the *same* initial noise + corrupted mask. PNGs only (personal review).

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python sweep_ddim_steps.py \
      -c config/config_prostate_v2.json \
      -e config/env_prostate_v2.json \
      --out-dir output/prostate_v2/ddim_sweep
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

from nnqc.corruptions import corrupt_ohe_masks_v2
from nnqc.utils import compute_spacing, define_instance, prepare_general_dataloader, prepare_msd_dataloader
from nnqc.xa import CLIPCrossAttentionGrid


def parse_args():
    p = argparse.ArgumentParser(description="DDIM step sweep (PNGs only, no TB)")
    p.add_argument("-c", "--config-file", default="./config/config_prostate_v2.json")
    p.add_argument("-e", "--environment-file", default="./config/env_prostate_v2.json")
    p.add_argument("--checkpoint-suffix", choices=["last", "best"], default="last")
    p.add_argument("--num-volumes", type=int, default=3)
    p.add_argument("--steps-list", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--out-dir", type=str, default=None,
                   help="override output dir (default: <output_dir>/ddim_sweep)")
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

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_dir) / "ddim_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing PNGs to {out_dir}")

    # --- val dataloader (no caching) ---
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

    # --- scale factor (recompute like training) ---
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

    steps_list = sorted(set(int(s) for s in args.steps_list))

    for vol_idx, batch in enumerate(val_loader):
        if vol_idx >= args.num_volumes:
            break
        images = batch["label"].to(device)
        if args.num_classes > 1:
            images = ohe(images)
        scans = batch["image"].to(device).float()
        slice_ratios = batch["slice_label"].float().to(device)
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
            # one corrupted mask + one initial noise - reused across all step counts
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
            # deterministic initial noise so step-count is the only varying factor
            gen = torch.Generator(device=device).manual_seed(args.seed + vol_idx)
            z_init = torch.randn(z_enc.shape, generator=gen, device=device, dtype=z_enc.dtype)

            samples_by_steps = {}
            for n_steps in steps_list:
                z_t = z_init.clone()
                scheduler.set_timesteps(n_steps)
                for t in scheduler.timesteps:
                    t_b = torch.full((z_t.shape[0],), int(t.item()), device=device, dtype=torch.long)
                    eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=c)
                    z_t = scheduler.step(eps, t, z_t)[0]
                samples_by_steps[n_steps] = autoencoder.decode_stage_2_outputs(z_t / scale_factor)
                print(f"  vol {vol_idx} | DDIM {n_steps} steps done")

        # --- figure: rows = apex/mid/base, cols = scan | corrupted | GT | sample@5 | @10 | ... ---
        ncols = 3 + len(steps_list)
        fig, axes = plt.subplots(3, ncols, figsize=(2.4 * ncols, 7.5))
        for row, (name, idx) in enumerate(picks):
            scan_img = scans_s[row, 0].cpu().numpy()
            if args.num_classes > 1:
                corr_lbl = corr_mask[row].argmax(0).cpu().numpy()
                gt_lbl = images_s[row].argmax(0).cpu().numpy()
                kw = dict(cmap="tab10", vmin=0, vmax=args.num_classes - 1)
            else:
                corr_lbl = corr_mask[row, 0].cpu().numpy()
                gt_lbl = images_s[row, 0].cpu().numpy()
                kw = dict(cmap="gray", vmin=0, vmax=1)

            axes[row, 0].imshow(scan_img, cmap="gray")
            axes[row, 1].imshow(corr_lbl, **kw)
            axes[row, 2].imshow(gt_lbl, **kw)
            for col, n_steps in enumerate(steps_list):
                sample = samples_by_steps[n_steps]
                if args.num_classes > 1:
                    samp_lbl = sample[row].argmax(0).cpu().numpy()
                else:
                    samp_lbl = (sample[row, 0].sigmoid() > 0.5).cpu().numpy()
                axes[row, 3 + col].imshow(samp_lbl, **kw)

            axes[row, 0].set_ylabel(f"{name}\nr={slice_ratios[idx].item():.2f}", fontsize=10)
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].set_title("scan")
        axes[0, 1].set_title("corrupted")
        axes[0, 2].set_title("GT")
        for col, n_steps in enumerate(steps_list):
            axes[0, 3 + col].set_title(f"DDIM {n_steps}")

        fig.suptitle(f"nnQC v2 - vol {vol_idx} - DDIM step sweep ({args.checkpoint_suffix})", y=1.0)
        fig.tight_layout()

        png_path = out_dir / f"vol{vol_idx:02d}_sweep.png"
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  vol {vol_idx}: saved {png_path.name}")

    print(f"Done. Sweep PNGs in {out_dir}")


if __name__ == "__main__":
    main()
