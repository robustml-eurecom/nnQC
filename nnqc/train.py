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
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.losses.dice import DiceCELoss, GeneralizedDiceLoss
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import DDIMScheduler
from monai.transforms import AsDiscrete as OHE
from monai.utils import first, progress_bar, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from nnqc.config import resolve_config
from nnqc.corruptions import corrupt_ohe_masks_v2, scaled_config
from nnqc.utils import (
    KL_loss,
    compute_spacing,
    define_instance,
    onehot_conditioning_enabled,
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


def _sample_corruption_cfg(rng_lo: float, rng_hi: float):
    """Draw a corruption severity for this step.

    Training previously corrupted every sample at exactly one severity
    (`corruption_prob=1.0`, default config), giving candidates in a ~0.02-wide
    Dice band - 0.811 +/- 0.02 on cardiac. A QC model is asked at inference to
    judge masks anywhere from broken to perfect, so both ends of its job were
    out of distribution: a clean mask (Dice 1.0) had never been seen, and neither
    had a badly degraded one.

    Sampling the severity per step covers the range instead. Note this is *not* a
    warmup schedule that ramps corruption up over epochs - that would make clean
    masks in-distribution early and out again later. Drawing uniformly from the
    start keeps the whole range live throughout.
    """
    # Degenerate range: return without touching the RNG. Drawing a variate here
    # would advance the global `random` stream that corrupt_binary_2d samples
    # from, so a "disabled" curriculum would still perturb every corruption
    # decision downstream - and a resumed run would diverge from the run it
    # resumes. Verified bit-identical to the pre-curriculum path.
    if rng_lo == rng_hi:
        return (scaled_config(rng_lo) if rng_lo != 1.0 else None), rng_lo
    import random as _random
    scale = _random.uniform(rng_lo, rng_hi)
    return scaled_config(scale), scale


def _ae_fp32(fn, *args):
    """Run an autoencoder call in fp32 even inside an ambient autocast region.

    The autoencoder is trained in fp32 (``_run_autoencoder`` uses no autocast)
    with ``norm_eps=1e-6``. That epsilon is **subnormal in fp16**, so under
    autocast a normalisation group whose input is constant - which happens
    whenever a class is absent from a slice, routine in 4-class ACDC - computes
    ``0 / sqrt(0 + 0)`` and yields NaN.

    Observed on cardiac: 6 of 32 slices came back entirely NaN, `scale_factor`
    became NaN, every loss followed, and since ``nan < best_val`` is always
    False the run trained for 3.5 h on 4 GPUs without ever saving a best
    checkpoint. fp32 costs some activation memory here and is worth it; the
    encoder calls are under ``no_grad`` anyway, and the decoder call stays
    differentiable so gradients still reach the UNet.
    """
    with autocast("cuda", enabled=False):
        return fn(*(a.float() if torch.is_tensor(a) else a for a in args))


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
    spacing = compute_spacing(cfg.data_base_dir, cfg, save=True)

    if cfg.is_msd:
        train_loader, val_loader = prepare_msd_dataloader(
            cfg, cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"],
            spacing=spacing, sample_axis=cfg.sample_axis, randcrop=True, rank=rank,
            world_size=world_size, cache=1.0, download=cfg.download, size_divisible=size_divisible,
        )
    else:
        train_loader, val_loader, _ = prepare_general_dataloader(
            cfg, cfg.image_pattern, cfg.label_pattern,
            cfg.autoencoder_train["batch_size"], cfg.autoencoder_train["patch_size"],
            spacing=spacing, sample_axis=cfg.sample_axis, randcrop=True,
            world_size=world_size, cache=1.0, size_divisible=size_divisible,
        )

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
    spacing = compute_spacing(cfg.data_base_dir, cfg, save=True)

    # Use the *diffusion* stage's batch/patch size. These used to be read from
    # `autoencoder_train`, so --batch-size / --patch-size on train-diffusion
    # (which config.py routes into diffusion_train) never reached the loader.
    dl_batch = dt.get("batch_size", cfg.autoencoder_train["batch_size"])
    dl_patch = dt.get("patch_size", cfg.autoencoder_train["patch_size"])
    if rank == 0:
        print(f"[nnqc] diffusion loader: batch_size={dl_batch} patch_size={dl_patch}")

    if cfg.is_msd:
        train_loader, val_loader = prepare_msd_dataloader(
            cfg, dl_batch, dl_patch, spacing,
            sample_axis=cfg.sample_axis, randcrop=True, rank=rank, world_size=world_size,
            cache=1.0, download=cfg.download, size_divisible=size_divisible,
        )
    else:
        train_loader, val_loader, _ = prepare_general_dataloader(
            cfg, cfg.image_pattern, cfg.label_pattern,
            dl_batch, dl_patch, spacing,
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
        z = _ae_fp32(autoencoder.encode_stage_2_inputs, check)
        if rank == 0:
            print(f"Latent feature shape {z.shape}")
    scale_factor = 1 / torch.std(z)
    # Fail loudly instead of training on NaN. A non-finite scale_factor poisons
    # every loss, and because `nan < best_val` is always False the best-checkpoint
    # branch never fires - so the run looks alive, writes `_last` checkpoints and
    # produces nothing usable. Cardiac burned 3.5 h on 4 GPUs exactly this way.
    if not torch.isfinite(scale_factor):
        raise RuntimeError(
            f"scale_factor is {scale_factor.item()} (std(z)={torch.std(z).item()}). "
            f"The autoencoder produced {int(torch.isnan(z).sum())} NaN / "
            f"{int(torch.isinf(z).sum())} Inf latent values from "
            f"{cfg.model_dir}/autoencoder.pt. Run "
            f"`python scripts/diagnose_scale_factor.py --task <task>` to localise it."
        )
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: scale_factor -> {scale_factor.item():.4f}")

    # Persist it next to the checkpoints. The latent scaling is part of the
    # trained model: evaluate() and check() used to re-estimate it from whatever
    # batch they happened to see (check() even estimated it from the *candidate
    # mask*), so sampling ran at a different scale than training.
    if rank == 0:
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.model_dir, "scale_factor.txt"), "w") as f:
            f.write(f"{scale_factor.item():.8f}")

    unet = define_instance(cfg, "diffusion_def").to(device)
    # Opt-in candidate-state gate (see xa.CLIPCrossAttentionGrid). Off by
    # default: the state_dict and the 2-token context are then unchanged.
    xa_gate = bool(cfg.diffusion_train.get("xa_mask_gate", False))
    # Per-class one-hot conditioning channels (area-resized) when the UNet's
    # in_channels asks for them; legacy single ordinal channel otherwise.
    # See utils.onehot_conditioning_enabled.
    onehot_cond = onehot_conditioning_enabled(cfg)
    xa = CLIPCrossAttentionGrid(
        output_dim=cfg.diffusion_def["cross_attention_dim"], grid_reduction="column_softmax",
        mask_gate=xa_gate,
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

    p_epoch_sidecar = os.path.join(cfg.model_dir, "diffusion_last_epoch.txt")

    start_epoch = 0
    if cfg.resume_ckpt:
        start_epoch = cfg.start_epoch
        # Prefer the epoch the previous leg actually reached. Batch campaigns
        # chain fixed-length legs, so a hand-supplied --start-epoch is only ever
        # an estimate; guessing high silently skips epochs and desynchronises
        # the LR schedule from the weights being resumed.
        if os.path.exists(p_epoch_sidecar):
            with open(p_epoch_sidecar) as f:
                recorded = int(float(f.read().strip()))
            if recorded != start_epoch:
                print(f"Rank {rank}: resuming at epoch {recorded} from {p_epoch_sidecar} "
                      f"(requested --start-epoch {start_epoch})")
            start_epoch = recorded
        r_unet = p_unet_last if os.path.exists(p_unet_last) else p_unet
        r_xa = p_xa_last if os.path.exists(p_xa_last) else p_xa
        r_embed = p_embed_last if os.path.exists(p_embed_last) else p_embed
        try:
            unet.load_state_dict(torch.load(r_unet, map_location=map_location, weights_only=True))
            # A gate-enabled model resuming from a pre-gate checkpoint has new
            # (untrained) mask_state weights that the file cannot supply.
            _xa_sd = torch.load(r_xa, map_location=map_location, weights_only=True)
            if xa_gate and not any(k.startswith("mask_state.") for k in _xa_sd):
                xa.load_state_dict(_xa_sd, strict=False)
                if rank == 0:
                    print("[nnqc] mask-state gate is new to this run; its weights start fresh.")
            else:
                xa.load_state_dict(_xa_sd)
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
    # Corruption severity range. Default (1.0, 1.0) reproduces the historical
    # single-severity behaviour exactly; set e.g. [0.0, 2.0] to span Dice ~0.45-1.0.
    _crange = dt.get("corruption_scale_range", [1.0, 1.0])
    corr_lo, corr_hi = float(_crange[0]), float(_crange[1])
    # Which validation scalar picks the best checkpoint. See the selection site
    # for the measurement that motivated changing the default: the total loss
    # chose cardiac epoch 91 over epoch 1990 weights that are 62% better.
    # Validation severity grid. Default [1.0] is the historical single-severity
    # pass, bit-identical RNG-wise. A wide-corruption run should set e.g.
    # [0.0, 1.0, 2.0]: with validation pinned at 1.0 only, the selector is
    # structurally blind to what wide training improves - measured on cardiac,
    # it preferred the severity-1.0 specialist (rho 0.83) over generalist
    # weights (rho 0.90) for the entire 40 h run. Fixed severities, never
    # sampled, so the number stays comparable across epochs.
    val_severities = [float(x) for x in dt.get("val_severities", [1.0])]
    best_metric = str(dt.get("best_metric", "recon")).lower()
    if best_metric not in ("recon", "total"):
        raise ValueError(f"diffusion_train.best_metric must be 'recon' or 'total', got {best_metric!r}")
    if rank == 0:
        print(f"[nnqc] corruption severity scale ~ U({corr_lo}, {corr_hi})"
              + ("  (single severity - historical behaviour)" if corr_lo == corr_hi else ""))
        print(f"[nnqc] best checkpoint selected on: {best_metric}")
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = start_epoch * len(train_loader)
    # The corruption range the stored `best_val` was earned under. Validation is
    # pinned to severity 1.0 (see compute_val_loss), so val_loss is comparable
    # across epochs *within* a regime - but a model trained on U(0, 2) is a
    # generalist and will usually score worse at severity 1.0 than the specialist
    # that produced the stored best. Carrying `best_val` across that change would
    # mean `ema_val < best_val` never fires again: training runs for days, and
    # `checkpoint="best"` silently keeps loading the pre-change model. So a
    # changed range invalidates the sidecar and the new regime earns its own best.
    #
    # `best_metric` is part of the same regime: `recon` and `total` are different
    # scalars on different scales (0.02 vs 0.26), so comparing a stored best from
    # one against the other is meaningless in both directions.
    range_sidecar = os.path.join(cfg.model_dir, "diffusion_corruption_range.txt")
    prev_range = None
    if os.path.exists(range_sidecar):
        with open(range_sidecar) as f:
            prev_range = f.read().strip()
    cur_range = f"{corr_lo},{corr_hi}|{best_metric}"
    if val_severities != [1.0]:
        cur_range += "|val=" + ",".join(f"{v:g}" for v in val_severities)
    if xa_gate:
        cur_range += "|gate"
    if onehot_cond:
        cur_range += "|onehot"

    best_val = 100.0
    if cfg.resume_ckpt and os.path.exists(best_val_sidecar):
        if prev_range is not None and prev_range != cur_range:
            _tag = prev_range.replace(",", "-").replace("|", "_")
            if rank == 0:
                print(f"[nnqc] corruption range changed {prev_range} -> {cur_range}; "
                      f"discarding best_val sidecar so the new regime earns its own best. "
                      f"The previous best checkpoint is preserved as *_sev{_tag}.pt.")
                for src, name in ((p_unet, "diffusion_unet"), (p_xa, "xa"), (p_embed, "embed")):
                    if os.path.exists(src):
                        shutil.copyfile(src, os.path.join(
                            cfg.model_dir, f"{name}_sev{_tag}.pt"))
        else:
            with open(best_val_sidecar) as f:
                best_val = float(f.read().strip())
            if rank == 0:
                print(f"Resumed best_val={best_val:.4f} from sidecar; best checkpoint protected.")
    if rank == 0:
        with open(range_sidecar, "w") as f:
            f.write(cur_range)

    train_loss_ema = train_loss_1_ema = train_loss_2_ema = None
    ema_alpha = 0.99
    loss_recon = GeneralizedDiceLoss(sigmoid=True)
    loss_2 = 0.0

    def compute_val_loss(epoch):
        # Accumulate l1/l2 over the whole val set, not just the last batch.
        # `last_l2` used to hold a single batch's value and was never all-reduced,
        # which was harmless while it was only a TensorBoard curve - but it now
        # selects the best checkpoint, and a one-batch, rank-0-only number is far
        # too noisy and DDP-inconsistent for that job.
        val_loss_sum = 0.0
        l1_sum = torch.tensor(0.0, device=device)
        l2_sum = torch.tensor(0.0, device=device)
        n = 0
        with torch.no_grad(), autocast("cuda", enabled=True):
            for step, batch in enumerate(val_loader):
                if step > 50:
                    break
                images = batch["label"].to(device)
                if cfg.num_classes > 1:
                    images = ohe(images)
                scans = batch["image"].to(device).float()
                slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)
                # NB: validation deliberately does NOT sample the curriculum - it is
                # pinned to a FIXED severity grid (default [1.0]). `val_loss` has exactly one load-bearing job,
                # selecting the best checkpoint, and that requires it to be comparable
                # across epochs and across a config change. Sampling U(lo, hi) here
                # would make each epoch's number a draw from a different distribution,
                # so `ema_val < best_val` would compare a lucky-easy epoch against a
                # historical fixed-severity value - either overwriting the best
                # checkpoint with a worse model or, since the resumed `best_val` comes
                # from the sidecar, freezing it forever. Train wide, validate fixed.
                vl_b = 0.0
                vl1_b = torch.tensor(0.0, device=device)
                vl2_b = torch.tensor(0.0, device=device)
                for _vsev in val_severities:
                    _ccfg_val, _ = _sample_corruption_cfg(_vsev, _vsev)
                    corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.0, config=_ccfg_val)
                    if onehot_cond:
                        # gate needs a foreground mask (label map, bg=0) so its
                        # fg fraction matches the legacy/binary call sites; the
                        # one-hot stack itself would read 1/C at every fill level.
                        gate_mask = corr_mask.argmax(1, keepdim=True)
                        cond = corr_mask.float()
                    elif cfg.num_classes > 1:
                        corr_mask = corr_mask.argmax(1, keepdim=True) / cfg.num_classes
                        gate_mask = cond = corr_mask
                    else:
                        gate_mask = cond = corr_mask
                    slice_emb = embed(slice_ratios).float().to(device)
                    c = xa.build_context(scans, slice_emb, mask=gate_mask).float().to(device)
                    noise_shape = [images.shape[0]] + list(z.shape[1:])
                    true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                    mask_resized = F.interpolate(cond, size=z.shape[2:],
                                                 mode="area" if onehot_cond else "nearest")
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                              (images.shape[0],), device=device).long()
                    ae = autoencoder.module if ddp_bool else autoencoder
                    z_enc = _ae_fp32(ae.encode_stage_2_inputs, images) * scale_factor
                    noisy_z = scheduler.add_noise(original_samples=z_enc, noise=true_noise, timesteps=timesteps)
                    noise_pred = unet(torch.cat([noisy_z, mask_resized], dim=1), timesteps=timesteps, context=c)
                    vl1 = F.mse_loss(noise_pred.float(), true_noise.float())
                    vl_b = vl_b + vl1
                    vl1_b = vl1_b + vl1.detach()
                    if epoch >= warmup_dice_epochs:
                        alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                        x0 = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                        decoded = _ae_fp32(ae.decode_stage_2_outputs, x0 / scale_factor)
                        vl2 = lambda_recon * loss_recon(decoded.float(), images.float())
                        vl_b = vl_b + vl2
                        vl2_b = vl2_b + vl2.detach()
                k = len(val_severities)
                val_loss_sum = val_loss_sum + vl_b / k
                l1_sum = l1_sum + vl1_b / k
                l2_sum = l2_sum + vl2_b / k
                n = step + 1
        val_loss_sum = val_loss_sum / max(n, 1)
        l1_sum = l1_sum / max(n, 1)
        l2_sum = l2_sum / max(n, 1)
        if ddp_bool:
            dist.barrier()
            for _t in (val_loss_sum, l1_sum, l2_sum):
                dist.all_reduce(_t, op=torch.distributed.ReduceOp.AVG)
        return val_loss_sum.item(), l1_sum.item(), l2_sum.item()

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
            _ccfg, _ = _sample_corruption_cfg(corr_lo, corr_hi)
            corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.0, config=_ccfg)
            if onehot_cond:
                gate_mask = corr_mask.argmax(1, keepdim=True)   # fg fraction, like the other paths
                cond = corr_mask.float()
            elif cfg.num_classes > 1:
                corr_mask = corr_mask.argmax(1, keepdim=True) / cfg.num_classes
                gate_mask = cond = corr_mask
            else:
                gate_mask = cond = corr_mask
            slice_emb = embed(slice_ratios).float().to(device)
            c = xa.build_context(scans, slice_emb, mask=gate_mask).float().to(device)

            optimizer_diff.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=True):
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                mask_resized = F.interpolate(cond, size=z.shape[2:],
                                             mode="area" if onehot_cond else "nearest")
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                          (images.shape[0],), device=device).long()
                ae = autoencoder.module if ddp_bool else autoencoder
                with torch.no_grad():
                    z_enc = _ae_fp32(ae.encode_stage_2_inputs, images) * scale_factor
                noisy_z = scheduler.add_noise(original_samples=z_enc, noise=true_noise, timesteps=timesteps)
                noise_pred = unet(torch.cat([noisy_z, mask_resized], dim=1), timesteps=timesteps, context=c)
                loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                loss = loss_1
                if epoch >= warmup_dice_epochs:
                    alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                    x0 = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                    seg_pred = _ae_fp32(ae.decode_stage_2_outputs, x0 / scale_factor)
                    loss_2 = lambda_recon * loss_recon(seg_pred.float(), images.float())
                    loss = loss + loss_2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_diff)
            # `unet` is DDP-wrapped and syncs its own gradients, but the trainable
            # cross-attention head and the slice embedding are plain modules: without
            # this all-reduce each rank would train its own divergent copy and only
            # rank 0's would ever be saved.
            if ddp_bool:
                for p in xa_trainable + embed_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
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
                lv = float(loss.detach())
                l1v = float(loss_1.detach())
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
            autoencoder.eval()
            unet.eval()
            embed.eval()
            xa.eval()
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
                # Record the epoch these `_last` weights correspond to, so a
                # resume leg picks up exactly here instead of trusting a
                # hand-supplied --start-epoch. Without it a chained campaign
                # re-trains everything between the guessed epoch and the real
                # one (and desynchronises the LR schedule from the weights).
                with open(p_epoch_sidecar, "w") as f:
                    f.write(str(epoch + 1))
                # Which scalar selects the best checkpoint. `total` (noise MSE +
                # lambda_recon * Dice) is dominated by the MSE term over randomly
                # sampled timesteps, and it demonstrably fails to see what we
                # actually care about: on cardiac it picked **epoch 91**, and the
                # epoch-1990 `_last` weights reconstruct at 0.598 Dice against
                # that checkpoint's 0.369 - a 62% improvement the selector scored
                # as worse, for 1900 epochs. `recon` selects on the decoded-mask
                # Dice term alone, which is the quantity the model is for.
                selector = ema_l2 if best_metric == "recon" else ema_val
                selector = float(selector.item() if torch.is_tensor(selector) else selector)
                # Before warmup the Dice term is identically 0, so it cannot rank
                # anything - fall back to the total loss until it goes live.
                if best_metric == "recon" and epoch < warmup_dice_epochs:
                    selector = float(ema_val)
                if selector < best_val:
                    best_val = selector
                    torch.save(unet_raw.state_dict(), p_unet)
                    torch.save(xa.state_dict(), p_xa)
                    torch.save(embed.state_dict(), p_embed)
                    with open(best_val_sidecar, "w") as f:
                        f.write(f"{best_val:.6f}")
                    print(f"Got best val ({best_metric}={selector:.5f}); saved to {p_unet}")
            unet_raw.load_state_dict(training_state)

        lr_scheduler.step()

    if rank == 0 and writer is not None:
        writer.flush()
        writer.close()
    return cfg.model_dir
