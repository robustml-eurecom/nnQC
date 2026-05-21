"""Training entry points for nnQC.

Two public functions, usable from a notebook or wrapped by the CLI:

    nnqc.train_autoencoder(task="prostate", epochs=500, lr=5e-5)
    nnqc.train_diffusion(task="prostate", epochs=4000, lr=2.5e-5,
                         scheduler="cosine", warmup_dice_epochs=100, resume=True)

Both accept either a bundled ``task=`` preset or an explicit
``config=<path>, env=<path>`` pair, plus keyword overrides for any field
(see :mod:`nnqc.config`). Execution knobs (``gpus``, ``device``, ``seed``)
are passed separately from model/data hyper-parameters.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.losses.dice import DiceCELoss, GeneralizedDiceLoss
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import DDIMScheduler
from monai.inferers import LatentDiffusionInferer
from monai.transforms import AsDiscrete as OHE
from monai.utils import first, progress_bar, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from nnqc.config import resolve_config
from nnqc.corruptions import corrupt_ohe_masks_v2
from nnqc.utils import (
    KL_loss,
    compute_spacing,
    define_instance,
    prepare_general_dataloader,
    prepare_msd_dataloader,
    setup_ddp,
)
from nnqc.visualize import visualize_2d_image
from nnqc.xa import CLIPCrossAttentionGrid


class EMA:
    """Exponential moving average of model weights.

    Diffusion sampling uses the averaged weights, which yields markedly
    smoother / more stable generations than the raw training weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        msd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(msd[k].dtype) for k in msd}, strict=True)

    def state_dict(self):
        return self.shadow


def _resolve_device(gpus, device):
    """Return (rank, world_size, device_index, ddp_bool, dist)."""
    if gpus is not None and gpus > 1:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, dev = setup_ddp(rank, world_size)
        torch.cuda.set_device(dev)
        return rank, world_size, dev, True, dist
    if device is None:
        dev = 0
    elif isinstance(device, int):
        dev = device
    elif isinstance(device, str):
        dev = 0 if device == "cuda" else int(device.split(":")[-1])
    else:
        dev = int(device)
    torch.cuda.set_device(dev)
    return 0, 1, dev, False, None


def _build_lr_scheduler(optimizer, name, total_epochs, warmup_epochs):
    """Map a scheduler name to a torch LR scheduler.

    Supported: ``cosine`` (linear warmup -> cosine anneal, the default),
    ``constant``, ``step``, ``exponential``.
    """
    name = (name or "cosine").lower()
    warmup_epochs = max(1, int(warmup_epochs))
    if name in ("cosine", "warmup_cosine"):
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
                ),
            ],
            milestones=[warmup_epochs],
        )
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _e: 1.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 3), gamma=0.5
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    raise ValueError(
        f"Unknown scheduler {name!r}. Choose from: cosine, constant, step, exponential."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train_autoencoder(
    config=None,
    env=None,
    task=None,
    *,
    gpus: int = 1,
    device=None,
    seed: int = 42,
    **overrides,
):
    """Train the AutoencoderKL that compresses one-hot masks into a latent.

    Examples
    --------
    >>> import nnqc
    >>> nnqc.train_autoencoder(task="prostate", epochs=500, lr=5e-5)
    >>> nnqc.train_autoencoder(config="configs/spleen/config.json",
    ...                        env="configs/spleen/env.json", device="cuda:2")
    """
    cfg = resolve_config(config, env, task, stage="autoencoder", overrides=overrides)
    return _run_autoencoder(cfg, gpus=gpus, device=device, seed=seed)


def train_diffusion(
    config=None,
    env=None,
    task=None,
    *,
    gpus: int = 1,
    device=None,
    seed: int = 42,
    **overrides,
):
    """Train the conditional diffusion UNet on the mask latent.

    Requires ``<model_dir>/autoencoder.pt`` to already exist.

    Examples
    --------
    >>> import nnqc
    >>> nnqc.train_diffusion(task="prostate", epochs=4000, lr=2.5e-5,
    ...                      scheduler="cosine", warmup_dice_epochs=100)
    >>> nnqc.train_diffusion(task="prostate", resume=True, start_epoch=1000,
    ...                      epochs=4000)
    """
    cfg = resolve_config(config, env, task, stage="diffusion", overrides=overrides)
    return _run_diffusion(cfg, gpus=gpus, device=device, seed=seed)


# --------------------------------------------------------------------------- #
# Autoencoder training loop
# --------------------------------------------------------------------------- #
def _run_autoencoder(cfg, *, gpus, device, seed):
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    rank, world_size, device, ddp_bool, dist = _resolve_device(gpus, device)
    print(f"[nnqc] autoencoder | device={device} ddp={ddp_bool} world_size={world_size}")
    set_determinism(seed)

    size_divisible = 2 ** (len(cfg.autoencoder_def["channels"]) - 1)
    spacing = compute_spacing("dataset", cfg, save=True)

    if cfg.is_msd:
        train_loader, val_loader = prepare_msd_dataloader(
            cfg, cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"],
            spacing=spacing, sample_axis=cfg.sample_axis, randcrop=True, rank=rank,
            world_size=world_size, cache=1.0, download=cfg.download, size_divisible=size_divisible,
        )
    else:
        _loaders = prepare_general_dataloader(
            cfg, cfg.image_pattern, cfg.label_pattern,
            cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"],
            spacing=spacing, sample_axis=cfg.sample_axis, randcrop=True,
            world_size=world_size, cache=1.0, size_divisible=size_divisible,
        )
        train_loader, val_loader = _loaders[0], _loaders[1]

    autoencoder = define_instance(cfg, "autoencoder_def").to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=cfg.spatial_dims, num_layers_d=3, channels=32,
        in_channels=cfg.num_classes, out_channels=1, norm="INSTANCE",
    ).to(device)
    if ddp_bool:
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    g_path = os.path.join(cfg.model_dir, "autoencoder.pt")
    d_path = os.path.join(cfg.model_dir, "discriminator.pt")
    g_path_last = os.path.join(cfg.model_dir, "autoencoder_last.pt")
    d_path_last = os.path.join(cfg.model_dir, "discriminator_last.pt")
    if rank == 0:
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    if cfg.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(g_path, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: loaded autoencoder from {g_path}")
        except Exception:
            print(f"Rank {rank}: train autoencoder from scratch.")
        try:
            discriminator.load_state_dict(torch.load(d_path, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: loaded discriminator from {d_path}")
        except Exception:
            print(f"Rank {rank}: train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=True)

    recon_loss = cfg.autoencoder_train.get("recon_loss")
    if recon_loss == "l2":
        intensity_loss = MSELoss()
    elif recon_loss == "dice_ce":
        intensity_loss = DiceCELoss(include_background=True, to_onehot_y=False,
                                    softmax=False, sigmoid=True, batch=True)
    else:
        intensity_loss = L1Loss()
    if rank == 0:
        print(f"[nnqc] reconstruction loss: {recon_loss or 'l1'}")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=cfg.spatial_dims, network_type="squeeze").to(device)
    adv_weight = 0.5
    perceptual_weight = cfg.autoencoder_train["perceptual_weight"]
    kl_weight = cfg.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=cfg.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.autoencoder_train["lr"] * world_size)

    writer = None
    if rank == 0:
        tb = os.path.join(cfg.tfevent_path, "autoencoder")
        Path(tb).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb)

    ohe = OHE(to_onehot=cfg.num_classes, dim=1)
    warm_up_epochs = 5
    max_epochs = cfg.autoencoder_train["max_epochs"]
    val_interval = cfg.autoencoder_train["val_interval"]
    best_val = 100.0
    total_step = 0
    recons_loss = 0.0

    print("\n[nnqc] start training autoencoder...")
    for epoch in range(max_epochs):
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            progress_bar(step, len(train_loader), f"epoch {epoch}, recon {recons_loss if step > 1 else 0:.4f}")
            images = batch["label"].to(device)
            if cfg.num_classes > 1:
                images = ohe(images)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            if cfg.num_classes > 1:
                p_loss = loss_perceptual(
                    F.softmax(reconstruction, dim=1).argmax(1, keepdim=True).float(),
                    images.argmax(1, keepdim=True).float(),
                )
            else:
                p_loss = loss_perceptual((torch.sigmoid(reconstruction) > 0.5).float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > warm_up_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > warm_up_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = adv_weight * (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                optimizer_d.step()

            if rank == 0:
                total_step += 1
                writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
                writer.add_scalar("train_kl_loss_iter", kl_loss, total_step)
                writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)

        if epoch % val_interval == 0:
            autoencoder.eval()
            val_loss = 0.0
            step = 0
            for step, batch in enumerate(val_loader):
                images = batch["label"].to(device)
                if cfg.num_classes > 1:
                    images = ohe(images)
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    rl = intensity_loss(reconstruction.float(), images.float())
                    if cfg.num_classes > 1:
                        p_loss = loss_perceptual(
                            F.softmax(reconstruction, dim=1).argmax(1, keepdim=True).float(),
                            images.argmax(1, keepdim=True).float(),
                        )
                    else:
                        p_loss = loss_perceptual((torch.sigmoid(reconstruction) > 0.5).float(), images.float())
                    rl += kl_weight * KL_loss(z_mu, z_sigma) + perceptual_weight * p_loss
                val_loss += rl.item()
            val_loss /= step + 1

            if rank == 0:
                print(f"Epoch {epoch} val_loss: {val_loss:.4f}")
                ae_sd = autoencoder.module.state_dict() if ddp_bool else autoencoder.state_dict()
                d_sd = discriminator.module.state_dict() if ddp_bool else discriminator.state_dict()
                torch.save(ae_sd, g_path_last)
                torch.save(d_sd, d_path_last)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ae_sd, g_path)
                    torch.save(d_sd, d_path)
                    print(f"Got best val recon loss; saved to {g_path}")
                mid = images.shape[0] // 2
                writer.add_scalar("val_recon_loss", val_loss, epoch)
                writer.add_image("val_img", visualize_2d_image(
                    images[mid].argmax(0) if cfg.num_classes > 1 else images[mid, 0]).transpose([2, 1, 0]), epoch)
                writer.add_image("val_recon", visualize_2d_image(
                    F.softmax(reconstruction[mid], dim=0).argmax(0) if cfg.num_classes > 1
                    else (F.sigmoid(reconstruction[mid, 0]) > 0.5).float()).transpose([2, 1, 0]), epoch)

    if rank == 0 and writer is not None:
        writer.flush()
        writer.close()
    return cfg.model_dir


# --------------------------------------------------------------------------- #
# Diffusion training loop
# --------------------------------------------------------------------------- #
def _run_diffusion(cfg, *, gpus, device, seed):
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    rank, world_size, device, ddp_bool, dist = _resolve_device(gpus, device)
    print(f"[nnqc] diffusion | device={device} ddp={ddp_bool} world_size={world_size}")
    set_determinism(seed)

    dt = cfg.diffusion_train
    size_divisible = 2 ** (len(cfg.autoencoder_def["channels"]) + len(cfg.diffusion_def["channels"]) - 2)
    spacing = compute_spacing("dataset", cfg, save=True)

    if cfg.is_msd:
        train_loader, val_loader = prepare_msd_dataloader(
            cfg, cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"], spacing,
            sample_axis=cfg.sample_axis, randcrop=True, rank=rank, world_size=world_size,
            cache=1.0, download=cfg.download, size_divisible=size_divisible,
        )
    else:
        train_loader, val_loader, _ = prepare_general_dataloader(
            cfg, cfg.image_pattern, cfg.label_pattern,
            cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"], spacing,
            sample_axis=cfg.sample_axis, randcrop=True, world_size=world_size,
            cache=1.0, size_divisible=size_divisible,
        )

    writer = None
    if rank == 0:
        tb = os.path.join(cfg.tfevent_path, "diffusion")
        Path(tb).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb)

    autoencoder = define_instance(cfg, "autoencoder_def").to(device)
    g_path = os.path.join(cfg.model_dir, "autoencoder.pt")
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(g_path, map_location=map_location, weights_only=True))
    print(f"Rank {rank}: loaded autoencoder from {g_path}")

    ohe = OHE(to_onehot=cfg.num_classes, dim=1)
    with torch.no_grad(), autocast("cuda", enabled=True):
        check = first(train_loader)["label"].to(device)
        if cfg.num_classes > 1:
            check = ohe(check)
        z = autoencoder.encode_stage_2_inputs(check)
        if rank == 0:
            print(f"Latent feature shape {z.shape}")
    scale_factor = 1 / torch.std(z)
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: scale_factor -> {scale_factor.item():.4f}")

    unet = define_instance(cfg, "diffusion_def").to(device)
    xa = CLIPCrossAttentionGrid(
        output_dim=cfg.diffusion_def["cross_attention_dim"], grid_reduction="column_softmax"
    ).to(device)
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.GELU(),
        torch.nn.Linear(32, cfg.diffusion_def["cross_attention_dim"]),
    ).to(device)

    p_unet = os.path.join(cfg.model_dir, "diffusion_unet.pt")
    p_unet_last = os.path.join(cfg.model_dir, "diffusion_unet_last.pt")
    p_xa = os.path.join(cfg.model_dir, "xa.pt")
    p_xa_last = os.path.join(cfg.model_dir, "xa_last.pt")
    p_embed = os.path.join(cfg.model_dir, "embed.pt")
    p_embed_last = os.path.join(cfg.model_dir, "embed_last.pt")
    best_val_sidecar = os.path.join(cfg.model_dir, "diffusion_best_val.txt")

    start_epoch = 0
    if cfg.resume_ckpt:
        start_epoch = cfg.start_epoch
        r_unet = p_unet_last if os.path.exists(p_unet_last) else p_unet
        r_xa = p_xa_last if os.path.exists(p_xa_last) else p_xa
        r_embed = p_embed_last if os.path.exists(p_embed_last) else p_embed
        try:
            unet.load_state_dict(torch.load(r_unet, map_location=map_location, weights_only=True))
            xa.load_state_dict(torch.load(r_xa, map_location=map_location, weights_only=True))
            embed.load_state_dict(torch.load(r_embed, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: resumed diffusion from {r_unet} at epoch {start_epoch}.")
        except Exception:
            print(f"Rank {rank}: train diffusion from scratch.")

    ema_decay = float(dt.get("ema_decay", 0.999))
    ema_unet = EMA(unet, decay=ema_decay)

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=cfg.NoiseScheduler["beta_start"],
        beta_end=cfg.NoiseScheduler["beta_end"],
        clip_sample=cfg.NoiseScheduler["clip_sample"],
    )

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    for name, param in xa.named_parameters():
        param.requires_grad = not (name.startswith("unimedclip") or name.startswith("tokenizer"))

    param_groups = []
    xa_trainable = [p for p in xa.parameters() if p.requires_grad]
    if xa_trainable:
        param_groups.append({"params": xa_trainable, "lr": dt["lr"], "name": "cross_attention"})
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    if unet_params:
        param_groups.append({"params": unet_params, "lr": dt["lr"], "name": "unet",
                             "weight_decay": float(dt.get("weight_decay", 1e-6))})
    embed_params = list(embed.parameters())
    if embed_params:
        param_groups.append({"params": embed_params, "lr": dt.get("embed_lr", 2.5e-5), "name": "slice_embeddings"})
    all_trainable = xa_trainable + unet_params + embed_params

    optimizer_diff = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

    total_epochs = dt["max_epochs"]
    warmup_epochs = int(dt.get("warmup_epochs", 20))
    lr_scheduler = _build_lr_scheduler(optimizer_diff, dt.get("scheduler", "cosine"), total_epochs, warmup_epochs)
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_scheduler.step()
        print(f"Rank {rank}: fast-forwarded LR scheduler by {start_epoch} epochs; "
              f"LRs={[g['lr'] for g in optimizer_diff.param_groups]}")

    max_epochs = dt["max_epochs"]
    val_interval = dt["val_interval"]
    warmup_dice_epochs = dt["warmup_dice_epochs"]
    lambda_recon = float(dt.get("lambda_recon", 0.1))
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = start_epoch * len(train_loader)
    best_val = 100.0
    if cfg.resume_ckpt and os.path.exists(best_val_sidecar):
        with open(best_val_sidecar) as f:
            best_val = float(f.read().strip())
        if rank == 0:
            print(f"Resumed best_val={best_val:.4f} from sidecar; best checkpoint protected.")

    train_loss_ema = train_loss_1_ema = train_loss_2_ema = None
    ema_alpha = 0.99
    loss_recon = GeneralizedDiceLoss(sigmoid=True)
    loss_2 = 0.0

    def compute_val_loss(epoch):
        val_loss_sum, last_l1, last_l2, n = 0.0, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0
        with torch.no_grad(), autocast("cuda", enabled=True):
            for step, batch in enumerate(val_loader):
                if step > 50:
                    break
                images = batch["label"].to(device)
                if cfg.num_classes > 1:
                    images = ohe(images)
                scans = batch["image"].to(device).float()
                slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)
                corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.0)
                if cfg.num_classes > 1:
                    corr_mask = corr_mask.argmax(1, keepdim=True) / cfg.num_classes
                slice_emb = embed(slice_ratios).float().to(device)
                c, _, _ = xa(scans, ext_features=slice_emb)
                c = c.float().to(device).unsqueeze(1)
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                mask_resized = F.interpolate(corr_mask.float(), size=z.shape[2:], mode="nearest")
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                          (images.shape[0],), device=device).long()
                ae = autoencoder.module if ddp_bool else autoencoder
                z_enc = ae.encode_stage_2_inputs(images) * scale_factor
                noisy_z = scheduler.add_noise(original_samples=z_enc, noise=true_noise, timesteps=timesteps)
                noise_pred = unet(torch.cat([noisy_z, mask_resized], dim=1), timesteps=timesteps, context=c)
                vl1 = F.mse_loss(noise_pred.float(), true_noise.float())
                vl = vl1
                last_l1 = vl1
                if epoch >= warmup_dice_epochs:
                    alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                    x0 = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                    decoded = ae.decode_stage_2_outputs(x0 / scale_factor)
                    vl2 = lambda_recon * loss_recon(decoded.float(), images.float())
                    vl = vl + vl2
                    last_l2 = vl2
                val_loss_sum = val_loss_sum + vl
                n = step + 1
        val_loss_sum = val_loss_sum / max(n, 1)
        if ddp_bool:
            dist.barrier()
            dist.all_reduce(val_loss_sum, op=torch.distributed.ReduceOp.AVG)
        return val_loss_sum.item(), last_l1, last_l2

    for epoch in range(start_epoch, max_epochs):
        unet.train()
        embed.train()
        xa.train()
        xa.unimedclip.eval()
        if hasattr(xa, "tokenizer") and hasattr(xa.tokenizer, "eval"):
            xa.tokenizer.eval()
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        if epoch == warmup_dice_epochs:
            print("\n[nnqc] warmup done; enabling decoded-mask Dice loss.\n")

        for step, batch in enumerate(train_loader):
            progress_bar(step, len(train_loader),
                         f"epoch {epoch}, total {loss if step > 1 else 0:.4f}, "
                         f"noise {loss_1 if step > 1 else 0:.4f}, recon {loss_2 if step > 1 else 0:.4f}")
            images = batch["label"].to(device)
            if cfg.num_classes > 1:
                images = ohe(images)
            scans = batch["image"].to(device).float()
            slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)
            corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.0)
            if cfg.num_classes > 1:
                corr_mask = corr_mask.argmax(1, keepdim=True) / cfg.num_classes
            slice_emb = embed(slice_ratios).float().to(device)
            c = xa(scans, ext_features=slice_emb)[0].float().unsqueeze(1).to(device)

            optimizer_diff.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=True):
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                mask_resized = F.interpolate(corr_mask.float(), size=z.shape[2:], mode="nearest")
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                          (images.shape[0],), device=device).long()
                ae = autoencoder.module if ddp_bool else autoencoder
                with torch.no_grad():
                    z_enc = ae.encode_stage_2_inputs(images) * scale_factor
                noisy_z = scheduler.add_noise(original_samples=z_enc, noise=true_noise, timesteps=timesteps)
                noise_pred = unet(torch.cat([noisy_z, mask_resized], dim=1), timesteps=timesteps, context=c)
                loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                loss = loss_1
                if epoch >= warmup_dice_epochs:
                    alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                    x0 = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                    seg_pred = ae.decode_stage_2_outputs(x0 / scale_factor)
                    loss_2 = lambda_recon * loss_recon(seg_pred.float(), images.float())
                    loss = loss + loss_2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_diff)
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            scaler.step(optimizer_diff)
            scaler.update()
            ema_unet.update(unet.module if ddp_bool else unet)

            if rank == 0:
                total_step += 1
                writer.add_scalar("train_diffusion_loss_iter", loss, total_step)
                writer.add_scalar("train_diffusion_loss_iter_1", loss_1, total_step)
                if epoch >= warmup_dice_epochs:
                    writer.add_scalar("train_diffusion_loss_iter_2", loss_2, total_step)
                lv = float(loss.detach()); l1v = float(loss_1.detach())
                l2v = float(loss_2.detach()) if torch.is_tensor(loss_2) else float(loss_2)
                train_loss_ema = lv if train_loss_ema is None else ema_alpha * train_loss_ema + (1 - ema_alpha) * lv
                train_loss_1_ema = l1v if train_loss_1_ema is None else ema_alpha * train_loss_1_ema + (1 - ema_alpha) * l1v
                train_loss_2_ema = l2v if train_loss_2_ema is None else ema_alpha * train_loss_2_ema + (1 - ema_alpha) * l2v
                writer.add_scalar("train_diffusion_loss_iter_ema", train_loss_ema, total_step)

        if rank == 0 and train_loss_ema is not None:
            writer.add_scalar("train_diffusion_loss_ema", train_loss_ema, epoch + 1)
            writer.add_scalar("train_diffusion_loss_1_ema", train_loss_1_ema, epoch + 1)
            if epoch >= warmup_dice_epochs:
                writer.add_scalar("train_diffusion_loss_2_ema", train_loss_2_ema, epoch + 1)

        if epoch % val_interval == 0:
            autoencoder.eval(); unet.eval(); embed.eval(); xa.eval()
            unet_raw = unet.module if ddp_bool else unet
            raw_val, raw_l1, raw_l2 = compute_val_loss(epoch)
            training_state = {k: v.detach().clone() for k, v in unet_raw.state_dict().items()}
            ema_unet.copy_to(unet_raw)
            ema_val, ema_l1, ema_l2 = compute_val_loss(epoch)

            if rank == 0:
                writer.add_scalar("val_diffusion_loss", ema_val, epoch + 1)
                writer.add_scalar("val_diffusion_loss_ema", ema_val, epoch + 1)
                writer.add_scalar("val_diffusion_loss_raw", raw_val, epoch + 1)
                writer.add_scalar("val_diffusion_loss_1", ema_l1, epoch + 1)
                writer.add_scalar("val_diffusion_loss_1_raw", raw_l1, epoch + 1)
                if epoch >= warmup_dice_epochs:
                    writer.add_scalar("val_diffusion_loss_2", ema_l2, epoch + 1)
                    writer.add_scalar("val_diffusion_loss_2_raw", raw_l2, epoch + 1)
                print(f"Epoch {epoch} val -> EMA {ema_val:.4f} | raw {raw_val:.4f}")
                torch.save(unet_raw.state_dict(), p_unet_last)
                torch.save(xa.state_dict(), p_xa_last)
                torch.save(embed.state_dict(), p_embed_last)
                if ema_val < best_val:
                    best_val = ema_val
                    torch.save(unet_raw.state_dict(), p_unet)
                    torch.save(xa.state_dict(), p_xa)
                    torch.save(embed.state_dict(), p_embed)
                    with open(best_val_sidecar, "w") as f:
                        f.write(f"{best_val:.6f}")
                    print(f"Got best val; saved to {p_unet}")
            unet_raw.load_state_dict(training_state)

        lr_scheduler.step()

    if rank == 0 and writer is not None:
        writer.flush()
        writer.close()
    return cfg.model_dir
