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

import numpy as np
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.transforms import AsDiscrete as OHE
from monai.losses.dice import DiceCELoss, GeneralizedDiceLoss
from monai.config import print_config
from monai.utils import first, set_determinism, progress_bar

from nnQC.utils import (
    define_instance,
    setup_ddp,
    corrupt_ohe_masks,
    visualize_2d_image
)
from nnQC.models.xa import CLIPCrossAttentionGrid

import gc
torch.cuda.empty_cache()
gc.collect()


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
    spacing = (1.0, 1.0, 1.0) #compute_spacing('dataset', args, save=True)
    
    if args.is_msd:
        from utils import prepare_msd_dataloader
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
        from utils import prepare_general_dataloader
        train_loader, val_loader = prepare_general_dataloader(
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

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_writer = SummaryWriter(tensorboard_path)

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
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location, weights_only=True))
            xa.load_state_dict(torch.load(trained_xa_path, map_location=map_location, weights_only=True))
            embed.load_state_dict(torch.load(trained_embed_path, map_location=map_location, weights_only=True))
            print(
                f"\nRank {rank}: Load trained diffusion model from",
                trained_diffusion_path, ", cross attention grid from", trained_xa_path,
                "positional embedding from", trained_embed_path,
                f"starting from epoch {start_epoch}.\n"
            )
        except:
            print(f"\nRank {rank}: Train diffusion model from scratch.\n")

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
            'lr': 1e-4,
            'name': 'slice_embeddings'
        })

    # Create optimizer
    optimizer_diff = torch.optim.Adam(
        param_groups,
        betas=(0.9, 0.999)
    )
    

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_diff,
        gamma=0.998,
    )

    # Step 4: training
    
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_recon_epoch_loss = 100.0
    
    loss_recon = GeneralizedDiceLoss(sigmoid=True)
    
    loss_2 = 0.0
    
    template = "A portal-venous phase CT scan in the axial plane of the spleen."
    lambda_recon = .1
    
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
            
            corr_mask = corrupt_ohe_masks(images, corruption_prob=1., fp_prob=.8)
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
                noise = torch.cat([true_noise, mask_resized], dim=1)

                # Create timesteps
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                # Get model prediction
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder

                noise_pred, seg_pred = inferer(
                    inputs=images,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=c
                )

                loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                loss = loss_1
                
                if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                    loss_2 = lambda_recon * loss_recon(seg_pred.float(), images.float())
                    loss += loss_2
                #loss = loss_1

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step)
                tensorboard_writer.add_scalar(
                    "train_diffusion_loss_iter_1", loss_1, total_step
                )
                if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                    tensorboard_writer.add_scalar(
                        "train_diffusion_loss_iter_2", loss_2, total_step
                    )

        # validation
        if (epoch) % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            embed.eval()
            xa.eval()
            
            val_recon_epoch_loss = 0
            with torch.no_grad():
                with autocast("cuda", enabled=True):
                    # compute val loss
                    for step, batch in enumerate(val_loader):
                        if step > 50:
                            break
                        images = batch["label"].to(device)
                        if args.num_classes > 1:
                            images = ohe(images)
                
                        scans = batch["image"].to(device).float()
                        slice_ratios = batch["slice_label"].unsqueeze(1).float().to(device)
                        
                        text = [template]*scans.shape[0]
                        corr_mask = corrupt_ohe_masks(images, corruption_prob=1., fp_prob=.8, erosion_prob=.8)
                        if args.num_classes > 1:
                            # argmax and scale to [0, 1]
                            corr_mask = corr_mask.argmax(1, keepdim=True) / args.num_classes
                        
                        slice_embeddings = embed(slice_ratios).float().to(device)
                                        
                        c, _, _ = xa(scans, ext_features=slice_embeddings)
                        c = c.float().to(device).unsqueeze(1)
                        
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        true_noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                        
                        mask_resized = F.interpolate(corr_mask.float(), size=z.shape[2:], mode='nearest')
                        noise = torch.cat([true_noise, mask_resized], dim=1)

                        timesteps = torch.randint(
                            0,
                            inferer.scheduler.num_train_timesteps,
                            (images.shape[0],),
                            device=images.device,
                        ).long()

                        # Get model prediction
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        noise_pred, denoised_latent = inferer(
                            inputs=images,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            condition=c
                        )
                        val_loss_1 = F.mse_loss(noise_pred.float(), true_noise.float())
                        val_loss = val_loss_1
                        if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                            val_loss_2 = lambda_recon * loss_recon(denoised_latent.float(), images.float())
                            val_loss += val_loss_2
                        
                        val_recon_epoch_loss += val_loss
                    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

                    if ddp_bool:
                        dist.barrier()
                        dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)

                    val_recon_epoch_loss = val_recon_epoch_loss.item()

                    # write val loss and save best model
                    if rank == 0:
                        tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch + 1)
                        tensorboard_writer.add_scalar(
                            "val_diffusion_loss_1", val_loss_1, epoch + 1
                        )
                        if epoch >= args.diffusion_train["warmup_dice_epochs"]:
                            tensorboard_writer.add_scalar(
                                "val_diffusion_loss_2", val_loss_2, epoch + 1
                            )
                        print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss:.4f}")
                        
                        
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path_last)
                        torch.save(xa.state_dict(), trained_xa_path_last)
                        torch.save(embed.state_dict(), trained_embed_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                                torch.save(xa.state_dict(), trained_xa_path)
                                torch.save(embed.state_dict(), trained_embed_path)
                            print("Got best val noise pred loss.")
                            print(
                                "Save trained latent diffusion model to",
                                trained_diffusion_path,
                            )
                        
                        slice_selected = min(50, images.shape[0] - 1)
                        scheduler.set_timesteps(50)            
                        # visualize synthesized image
                        
                        if (epoch) % (val_interval) == 0:  
                            synthetic_images = inferer.sample(
                                input_noise=noise[slice_selected-1:slice_selected, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=c[slice_selected-1:slice_selected, ...]
                            )
                            tensorboard_writer.add_image(
                                "val_corrupted_mask",
                                visualize_2d_image(corr_mask[slice_selected, 0]).transpose([2, 1, 0]),
                                epoch,
                            )
                            tensorboard_writer.add_image(
                                "val_diff_synimg",
                                visualize_2d_image(F.softmax(synthetic_images[0]).argmax(dim=0) if args.num_classes > 1 else F.sigmoid(synthetic_images[0, 0])).transpose([2, 1, 0]),
                                epoch,
                            )
                            tensorboard_writer.add_image(
                                "val_denoised_latent",
                                visualize_2d_image(F.softmax(denoised_latent[0]).argmax(dim=0) if args.num_classes > 1 else F.sigmoid(denoised_latent[0, 0])).transpose([2, 1, 0]),
                                epoch,
                            )
                            tensorboard_writer.add_image(
                                "val_diff_mask",
                                visualize_2d_image(images[slice_selected].argmax(dim=0) if args.num_classes > 1 else images[slice_selected,0]).transpose([2, 1, 0]),
                                epoch,
                            )

        lr_scheduler.step()

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
