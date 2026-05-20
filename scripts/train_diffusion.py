# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.transforms import AsDiscrete as OHE
from monai.losses.dice import DiceCELoss, GeneralizedDiceLoss
from monai.config import print_config
from monai.utils import first, set_determinism, progress_bar
from torch.amp import autocast
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import wandb  # kept for compatibility; logging disabled below
from nnqc.utils import (
    define_instance,
    setup_ddp,
    compute_spacing
)
from nnqc.corruptions import corrupt_ohe_masks_v2
from nnqc.visualize import visualize_2d_image
from nnqc.xa import CLIPCrossAttentionGrid
import numpy as np

import gc
torch.cuda.empty_cache()
gc.collect()


class EMA:
    """Exponential moving average of model weights.

    Diffusion sampling uses the averaged weights, which yields markedly
    smoother / more stable generations than the raw training weights.
    """

    def __init__(self, model, decay=0.9999):
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # config printing (paths, dataset, model, etc.)
    print("=" * 50)
    print("Environment variables:")
    for k, v in env_dict.items():
        print(f"{k}: {v}")
    print("=" * 50)
    print()
    
    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    spacing = compute_spacing('dataset', args, save=True)
    
    if args.is_msd:
        from nnqc.utils import prepare_msd_dataloader
        train_loader, val_loader = prepare_msd_dataloader(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            spacing,
            sample_axis=args.sample_axis,
            randcrop=True,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            download=args.download,
            size_divisible=size_divisible,
        )
    else:
        from nnqc.utils import prepare_general_dataloader
        train_loader, val_loader, _ = prepare_general_dataloader(
            args,
            args.image_pattern,
            args.label_pattern,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            spacing,
            sample_axis=args.sample_axis,
            randcrop=True,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
        )

    # initialize tensorboard writer + wandb
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_writer = SummaryWriter(tensorboard_path)
        pass  # wandb disabled; using tensorboard only

    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
    logging.info(f"Rank {rank}: Load trained autoencoder from {trained_g_path}\n")
    
    ohe = OHE(to_onehot=args.num_classes, dim=1)
    
    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = first(train_loader)
            check_input = check_data["label"].to(device)
            if args.num_classes > 1:
                check_input = ohe(check_input)
                
            z = autoencoder.encode_stage_2_inputs(check_input)
            if rank == 0:
                print(f"Latent feature shape {z.shape}")
                tensorboard_writer.add_image(
                    "train_img",
                    visualize_2d_image(check_input[10,0]).transpose([2, 1, 0]),
                    1,
                )
                print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    print(f"Rank {rank}: local scale_factor: {scale_factor}")
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: final scale_factor -> {scale_factor}\n")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(device)
    xa = CLIPCrossAttentionGrid(output_dim=args.diffusion_def['cross_attention_dim'], grid_reduction='column_softmax').to(device)
    #embed = torch.nn.Embedding(num_embeddings=2, embedding_dim=512).to(device)
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), 
        torch.nn.GELU(), 
        torch.nn.Linear(32, args.diffusion_def['cross_attention_dim'])
        ).to(device)
    
    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")
    trained_xa_path = os.path.join(args.model_dir, "xa.pt")
    trained_xa_path_last = os.path.join(args.model_dir, "xa_last.pt")
    trained_embed_path = os.path.join(args.model_dir, "embed.pt")
    trained_embed_path_last = os.path.join(args.model_dir, "embed_last.pt")

    start_epoch = 0
    if args.resume_ckpt:
        start_epoch = args.start_epoch
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        # Prefer the "_last" checkpoint (most recent EMA weights) so resume picks
        # up where the prior run actually stopped, not at an earlier "best".
        resume_unet = trained_diffusion_path_last if os.path.exists(trained_diffusion_path_last) else trained_diffusion_path
        resume_xa = trained_xa_path_last if os.path.exists(trained_xa_path_last) else trained_xa_path
        resume_embed = trained_embed_path_last if os.path.exists(trained_embed_path_last) else trained_embed_path
        try:
            unet.load_state_dict(torch.load(resume_unet, map_location=map_location, weights_only=True))
            xa.load_state_dict(torch.load(resume_xa, map_location=map_location, weights_only=True))
            embed.load_state_dict(torch.load(resume_embed, map_location=map_location, weights_only=True))
            print(
                f"\nRank {rank}: Load trained diffusion model from",
                resume_unet, ", cross attention grid from", resume_xa,
                "positional embedding from", resume_embed,
                f"starting from epoch {start_epoch}.\n"
            )
        except:
            print(f"\nRank {rank}: Train diffusion model from scratch.\n")

    # EMA of the diffusion UNet - used for validation and saved checkpoints.
    # decay 0.999 (half-life ~28 epochs at 25 steps/epoch) suits this run's step budget;
    # 0.9999 would leave the EMA dominated by the random init for hundreds of epochs.
    ema_unet = EMA(unet, decay=0.999)

    scheduler = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=args.NoiseScheduler["clip_sample"],
    )

    if ddp_bool:
        autoencoder = DDP(
            autoencoder,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # Step 3: training config    
    #set xa trainable parameters
    for name, param in xa.named_parameters():
        if name.startswith("unimedclip") or name.startswith("tokenizer"):
            param.requires_grad = False
        else:
            param.requires_grad = True

    param_groups = []

    xa_trainable_params = [p for p in xa.parameters() if p.requires_grad]
    if xa_trainable_params:
        param_groups.append({
            'params': xa_trainable_params, 
            'lr': args.diffusion_train["lr"],  # Lower initial LR for cross-attention
            'name': 'cross_attention',
        })

    unet_params = [p for p in unet.parameters() if p.requires_grad]  # Replace with your UNet
    if unet_params:
        param_groups.append({
            'params': unet_params, 
            'lr': args.diffusion_train["lr"],  # Full LR for main model
            'name': 'unet',
            'weight_decay': 1e-6
        })

    embed_params = list(embed.parameters())
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': 2.5e-5,
            'name': 'slice_embeddings'
        })

    # all trainable params, for gradient clipping
    all_trainable_params = xa_trainable_params + unet_params + embed_params

    # Create optimizer
    optimizer_diff = torch.optim.Adam(
        param_groups,
        betas=(0.9, 0.999)
    )
    

    # warmup -> cosine annealing: smoother decay and more effective than plain ExponentialLR
    warmup_epochs = 20
    total_epochs = args.diffusion_train["max_epochs"]
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer_diff,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer_diff, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_diff, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
            ),
        ],
        milestones=[warmup_epochs],
    )

    # When resuming, fast-forward the scheduler so the resumed run picks up
    # on the cosine tail instead of restarting from the linear warmup.
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_scheduler.step()
        print(f"Rank {rank}: fast-forwarded LR scheduler by {start_epoch} epochs; "
              f"current LR(s): {[g['lr'] for g in optimizer_diff.param_groups]}")

    # Step 4: training
    
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    # EMA-smoothed running estimates of the train losses, logged once per epoch.
    train_loss_ema = None
    train_loss_1_ema = None
    train_loss_2_ema = None
    train_loss_ema_alpha = 0.99
    
    loss_recon = GeneralizedDiceLoss(sigmoid=True)
    
    loss_2 = 0.0

    lambda_recon = .1

    def compute_val_loss(epoch):
        """Run the validation loop with whatever weights `unet` currently holds.

        Returns (mean_total_loss, last_noise_loss, last_recon_loss).
        """
        val_recon_epoch_loss = 0
        last_l1 = torch.tensor(0.0, device=device)
        last_l2 = torch.tensor(0.0, device=device)
        n = 0
        with torch.no_grad():
            with autocast("cuda", enabled=True):
                for step, batch in enumerate(val_loader):
                    if step > 50:
                        break
                    images = batch["label"].to(device)
                    if args.num_classes > 1:
                        images = ohe(images)
                    scans = batch["image"].to(device).float()
                    slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)

                    # same corruption distribution as training (matched train/val)
                    corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.)
                    if args.num_classes > 1:
                        corr_mask = corr_mask.argmax(1, keepdim=True) / args.num_classes

                    slice_embeddings = embed(slice_ratios).float().to(device)
                    c, _, _ = xa(scans, ext_features=slice_embeddings)
                    c = c.float().to(device).unsqueeze(1)

                    noise_shape = [images.shape[0]] + list(z.shape[1:])
                    true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                    mask_resized = F.interpolate(corr_mask.float(), size=z.shape[2:], mode='nearest')

                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps,
                        (images.shape[0],), device=images.device,
                    ).long()

                    inferer_autoencoder = autoencoder.module if ddp_bool else autoencoder
                    z_enc = inferer_autoencoder.encode_stage_2_inputs(images) * scale_factor
                    noisy_z = scheduler.add_noise(
                        original_samples=z_enc, noise=true_noise, timesteps=timesteps
                    )
                    unet_input = torch.cat([noisy_z, mask_resized], dim=1)
                    noise_pred = unet(unet_input, timesteps=timesteps, context=c)

                    val_loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                    val_loss = val_loss_1
                    last_l1 = val_loss_1
                    if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                        alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                        x0_pred = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                        denoised_latent = inferer_autoencoder.decode_stage_2_outputs(x0_pred / scale_factor)
                        val_loss_2 = lambda_recon * loss_recon(denoised_latent.float(), images.float())
                        val_loss = val_loss + val_loss_2
                        last_l2 = val_loss_2

                    val_recon_epoch_loss += val_loss
                    n = step + 1

        val_recon_epoch_loss = val_recon_epoch_loss / max(n, 1)
        if ddp_bool:
            dist.barrier()
            dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)
        return val_recon_epoch_loss.item(), last_l1, last_l2

    for epoch in range(start_epoch, max_epochs):
        unet.train()
        embed.train()
        xa.train()
        
        xa.unimedclip.eval()
        if hasattr(xa, 'tokenizer') and hasattr(xa.tokenizer, 'eval'):
            xa.tokenizer.eval()
        
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        if epoch == args.diffusion_train["warmup_dice_epochs"]:
            print("\nFinish warmup epochs, start training with dice loss.\n")
        
        for step, batch in enumerate(train_loader):  
            #if step > 1:
            #    break   
            progress_bar(
                step,
                len(train_loader),
                f"epoch {epoch}, \
                    Total Loss: {loss if (step > 1) else 0:.4f}, \
                        Noise loss: {loss_1 if (step > 1) else 0:.4f}, \
                            Recon Loss: {loss_2 if (step > 1) else 0:.4f}",
            )
            
            images = batch["label"].to(device)
            if args.num_classes > 1:
                images = ohe(images)
                
            scans = batch["image"].to(device).float()
            slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)
            
            corr_mask = corrupt_ohe_masks_v2(images, corruption_prob=1.)
            if args.num_classes > 1:
                # argmax and scale to [0, 1]
                corr_mask = corr_mask.argmax(1, keepdim=True) / args.num_classes

            slice_embeddings = embed(slice_ratios).float().to(device)
            
            c = xa(scans, ext_features=slice_embeddings)[0].float().unsqueeze(1).to(device)
            
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                mask_resized = F.interpolate(corr_mask.float(), size=z.shape[2:], mode='nearest')

                # Create timesteps
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                # Get model prediction (manual forward pass: noise only has latent channels)
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder

                with torch.no_grad():
                    z_enc = inferer_autoencoder.encode_stage_2_inputs(images) * scale_factor
                noisy_z = scheduler.add_noise(
                    original_samples=z_enc, noise=true_noise, timesteps=timesteps
                )
                unet_input = torch.cat([noisy_z, mask_resized], dim=1)
                noise_pred = unet(unet_input, timesteps=timesteps, context=c)

                loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                loss = loss_1

                if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                    alpha_prod = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                    x0_pred = (noisy_z - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
                    seg_pred = inferer_autoencoder.decode_stage_2_outputs(x0_pred / scale_factor)
                    loss_2 = lambda_recon * loss_recon(seg_pred.float(), images.float())
                    loss += loss_2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_diff)
            torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
            scaler.step(optimizer_diff)
            scaler.update()
            ema_unet.update(unet.module if ddp_bool else unet)

            # write train loss for each batch into tensorboard + wandb
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step)
                tensorboard_writer.add_scalar("train_diffusion_loss_iter_1", loss_1, total_step)
                log_dict = {"train/loss": loss.item(), "train/noise_loss": loss_1.item(), "step": total_step}
                if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                    tensorboard_writer.add_scalar("train_diffusion_loss_iter_2", loss_2, total_step)
                    log_dict["train/recon_loss"] = loss_2.item()
                pass  # wandb.log disabled

                # exponential moving average of the scalar train losses
                loss_val = float(loss.detach())
                loss_1_val = float(loss_1.detach())
                loss_2_val = float(loss_2.detach()) if torch.is_tensor(loss_2) else float(loss_2)
                a = train_loss_ema_alpha
                train_loss_ema = loss_val if train_loss_ema is None else a * train_loss_ema + (1 - a) * loss_val
                train_loss_1_ema = loss_1_val if train_loss_1_ema is None else a * train_loss_1_ema + (1 - a) * loss_1_val
                train_loss_2_ema = loss_2_val if train_loss_2_ema is None else a * train_loss_2_ema + (1 - a) * loss_2_val
                tensorboard_writer.add_scalar("train_diffusion_loss_iter_ema", train_loss_ema, total_step)

        # end of epoch - log EMA-smoothed train losses once per epoch
        if rank == 0 and train_loss_ema is not None:
            tensorboard_writer.add_scalar("train_diffusion_loss_ema", train_loss_ema, epoch + 1)
            tensorboard_writer.add_scalar("train_diffusion_loss_1_ema", train_loss_1_ema, epoch + 1)
            if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                tensorboard_writer.add_scalar("train_diffusion_loss_2_ema", train_loss_2_ema, epoch + 1)

        # validation
        if (epoch) % val_interval == 0:
            # --- validation: evaluate both the raw model and the EMA model ---
            autoencoder.eval()
            unet.eval()
            embed.eval()
            xa.eval()
            unet_raw = unet.module if ddp_bool else unet

            # 1) raw (training) weights
            raw_val_loss, raw_l1, raw_l2 = compute_val_loss(epoch)

            # 2) EMA weights - swapped into the unet; this is what gets shipped/checkpointed
            training_unet_state = {k: v.detach().clone() for k, v in unet_raw.state_dict().items()}
            ema_unet.copy_to(unet_raw)
            ema_val_loss, ema_l1, ema_l2 = compute_val_loss(epoch)

            if rank == 0:
                # val_diffusion_loss kept == EMA loss (drives best-model selection)
                tensorboard_writer.add_scalar("val_diffusion_loss", ema_val_loss, epoch + 1)
                tensorboard_writer.add_scalar("val_diffusion_loss_ema", ema_val_loss, epoch + 1)
                tensorboard_writer.add_scalar("val_diffusion_loss_raw", raw_val_loss, epoch + 1)
                tensorboard_writer.add_scalar("val_diffusion_loss_1", ema_l1, epoch + 1)
                tensorboard_writer.add_scalar("val_diffusion_loss_1_raw", raw_l1, epoch + 1)
                if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                    tensorboard_writer.add_scalar("val_diffusion_loss_2", ema_l2, epoch + 1)
                    tensorboard_writer.add_scalar("val_diffusion_loss_2_raw", raw_l2, epoch + 1)
                print(f"Epoch {epoch} val_diffusion_loss -> EMA: {ema_val_loss:.4f} | raw: {raw_val_loss:.4f}")

                # save latest (unet currently holds EMA weights)
                torch.save(unet_raw.state_dict(), trained_diffusion_path_last)
                torch.save(xa.state_dict(), trained_xa_path_last)
                torch.save(embed.state_dict(), trained_embed_path_last)

                # save best model (by EMA val loss)
                if ema_val_loss < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = ema_val_loss
                    torch.save(unet_raw.state_dict(), trained_diffusion_path)
                    torch.save(xa.state_dict(), trained_xa_path)
                    torch.save(embed.state_dict(), trained_embed_path)
                    print("Got best val noise pred loss. Saved to", trained_diffusion_path)

            # restore raw training weights (EMA keeps accumulating separately)
            unet_raw.load_state_dict(training_unet_state)

        lr_scheduler.step()

    if rank == 0:
        pass  # wandb.finish disabled

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
