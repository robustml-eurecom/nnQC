"""Evaluation / visualization for a trained nnQC model.

Loads checkpoints from ``<model_dir>``, runs DDIM sampling on a few validation
volumes, and renders apex / mid / base reconstruction panels
(scan | corrupted input | GT | sampled reconstruction).

    nnqc.evaluate(task="prostate", num_volumes=3, num_steps=5)
"""
from __future__ import annotations

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

from nnqc.config import resolve_config
from nnqc.corruptions import corrupt_ohe_masks_v2
from nnqc.utils import compute_spacing, define_instance, prepare_general_dataloader, prepare_msd_dataloader
from nnqc.xa import CLIPCrossAttentionGrid


def evaluate(
    config=None,
    env=None,
    task=None,
    *,
    checkpoint="last",
    num_volumes: int = 3,
    num_steps: int = 5,
    step: int = 0,
    device=None,
    seed: int = 42,
    auto_download: bool = True,
    hf_token=None,
    hf_repo: str = "sanbast/nnQC",
    **overrides,
):
    """Sample reconstructions and write comparison panels.

    Parameters
    ----------
    checkpoint
        ``"last"`` (diffusion_unet_last.pt) or ``"best"`` (diffusion_unet.pt).
    num_volumes
        Number of validation volumes to render.
    num_steps
        DDIM sampling steps (5 is the recommended quality/latency trade-off).
    step
        TensorBoard step index (typically the train epoch).

    Returns the output directory containing the PNG panels.
    """
    cfg = resolve_config(config, env, task, stage="diffusion", overrides=overrides)
    set_determinism(seed)

    if device is None:
        dev = torch.device("cuda")
    else:
        dev = torch.device(device if isinstance(device, str) else f"cuda:{int(device)}")
    print(f"[nnqc] evaluate | device={dev}")

    out_dir = Path(cfg.output_dir) / f"eval_step_{step:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg.tfevent_path) / "eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    size_divisible = 2 ** (len(cfg.autoencoder_def["channels"]) + len(cfg.diffusion_def["channels"]) - 2)
    spacing = compute_spacing("dataset", cfg, save=False)
    if cfg.is_msd:
        _, val_loader = prepare_msd_dataloader(
            cfg, cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"], spacing,
            sample_axis=cfg.sample_axis, randcrop=True, world_size=1, cache=0.0,
            download=False, size_divisible=size_divisible,
        )
    else:
        _, val_loader, _ = prepare_general_dataloader(
            cfg, cfg.image_pattern, cfg.label_pattern,
            cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"], spacing,
            sample_axis=cfg.sample_axis, randcrop=True, world_size=1, cache=0.0,
            size_divisible=size_divisible,
        )

    ohe = OHE(to_onehot=cfg.num_classes, dim=1)

    if auto_download and task is not None:
        from nnqc.hub import ensure_weights
        ensure_weights(task, cfg.model_dir, token=hf_token, repo_id=hf_repo)

    autoencoder = define_instance(cfg, "autoencoder_def").to(dev).eval()
    autoencoder.load_state_dict(torch.load(
        os.path.join(cfg.model_dir, "autoencoder.pt"), map_location=dev, weights_only=True))

    unet = define_instance(cfg, "diffusion_def").to(dev).eval()
    ckpt = "diffusion_unet.pt" if checkpoint == "best" else "diffusion_unet_last.pt"
    unet.load_state_dict(torch.load(os.path.join(cfg.model_dir, ckpt), map_location=dev, weights_only=True))
    print(f"  loaded UNet from {ckpt}")

    xa = CLIPCrossAttentionGrid(
        output_dim=cfg.diffusion_def["cross_attention_dim"], grid_reduction="column_softmax",
    ).to(dev).eval()
    xa.load_state_dict(torch.load(
        os.path.join(cfg.model_dir, "xa.pt" if checkpoint == "best" else "xa_last.pt"),
        map_location=dev, weights_only=True))

    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.GELU(),
        torch.nn.Linear(32, cfg.diffusion_def["cross_attention_dim"]),
    ).to(dev).eval()
    embed.load_state_dict(torch.load(
        os.path.join(cfg.model_dir, "embed.pt" if checkpoint == "best" else "embed_last.pt"),
        map_location=dev, weights_only=True))

    with torch.no_grad(), autocast("cuda", enabled=True):
        check = first(val_loader)["label"].to(dev)
        if cfg.num_classes > 1:
            check = ohe(check)
        z = autoencoder.encode_stage_2_inputs(check)
        scale_factor = 1.0 / torch.std(z)
    print(f"  scale_factor = {scale_factor.item():.4f}")

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=cfg.NoiseScheduler["beta_start"],
        beta_end=cfg.NoiseScheduler["beta_end"],
        clip_sample=cfg.NoiseScheduler["clip_sample"],
    )

    logged = 0
    for vol_idx, batch in enumerate(val_loader):
        if vol_idx >= num_volumes:
            break
        images = batch["label"].to(dev)
        if cfg.num_classes > 1:
            images = ohe(images)
        scans = batch["image"].to(dev).float()
        slice_ratios = batch["slice_label"].float().to(dev)
        n = images.shape[0]
        if n < 3:
            print(f"  vol {vol_idx}: only {n} slices, skipped")
            continue

        order = torch.argsort(slice_ratios).tolist()
        picks = [("apex", order[0]), ("mid", order[len(order) // 2]), ("base", order[-1])]
        sub = [i for _, i in picks]
        images_s, scans_s = images[sub], scans[sub]
        ratios_s = slice_ratios[sub].unsqueeze(1)

        with torch.no_grad(), autocast("cuda", enabled=True):
            corr_mask = corrupt_ohe_masks_v2(images_s, corruption_prob=1.0)
            corr_in = (corr_mask.argmax(1, keepdim=True).float() / cfg.num_classes
                       if cfg.num_classes > 1 else corr_mask.float())
            slice_emb = embed(ratios_s).float()
            c, _, _ = xa(scans_s, ext_features=slice_emb)
            c = c.float().unsqueeze(1)
            z_enc = autoencoder.encode_stage_2_inputs(images_s) * scale_factor
            mask_resized = F.interpolate(corr_in, size=z_enc.shape[2:], mode="nearest")
            z_t = torch.randn(z_enc.shape, device=dev, dtype=z_enc.dtype)
            scheduler.set_timesteps(num_steps)
            for t in scheduler.timesteps:
                t_b = torch.full((z_t.shape[0],), int(t.item()), device=dev, dtype=torch.long)
                eps = unet(torch.cat([z_t, mask_resized], dim=1), timesteps=t_b, context=c)
                z_t = scheduler.step(eps, t, z_t)[0]
            sampled = autoencoder.decode_stage_2_outputs(z_t / scale_factor)

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for row, (name, idx) in enumerate(picks):
            scan_img = scans_s[row, 0].cpu().numpy()
            if cfg.num_classes > 1:
                corr_lbl = corr_mask[row].argmax(0).cpu().numpy()
                gt_lbl = images_s[row].argmax(0).cpu().numpy()
                samp_lbl = sampled[row].argmax(0).cpu().numpy()
                kw = dict(cmap="tab10", vmin=0, vmax=cfg.num_classes - 1)
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
        axes[0, 3].set_title(f"sample (DDIM {num_steps})")
        fig.suptitle(f"nnQC - volume {vol_idx} - step {step}", y=1.0)
        fig.tight_layout()
        png = out_dir / f"vol{vol_idx:02d}.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        writer.add_figure(f"eval/vol{vol_idx:02d}", fig, step)
        plt.close(fig)
        logged += 1
        print(f"  vol {vol_idx}: saved {png.name}")

    writer.flush()
    writer.close()
    print(f"[nnqc] done. {logged} figures in {out_dir}")
    return str(out_dir)
