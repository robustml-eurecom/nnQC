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
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.losses.dice import DiceCELoss
from monai.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism, progress_bar
from monai.transforms import AsDiscrete as OHE

from torch.utils.tensorboard import SummaryWriter
from nnQC.utils import KL_loss, define_instance, prepare_msd_dataloader, setup_ddp
from nnQC.utils import visualize_2d_image

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
torch.cuda.synchronize()

# Set cuDNN settings for stability
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # Change this to False
torch.backends.cudnn.deterministic = True


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
        rank = args.gpus
        world_size = 1
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
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    print("=" * 50)
    print("Environment variables:")
    for k, v in env_dict.items():
        print(f"{k}: {v}")
    print("=" * 50)
    print()
    
    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) - 1)
    spacing = (1.0, 1.0, 1.0)#compute_spacing('dataset', args, save=True)
    
    if args.is_msd:
        from utils import prepare_msd_dataloader
        train_loader, val_loader = prepare_msd_dataloader(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            spacing=spacing,
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
            spacing=spacing,
            sample_axis=args.sample_axis,
            randcrop=True,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
        )

    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=args.num_classes,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(
            autoencoder,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )
        discriminator = DDP(
            discriminator,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    elif "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "dice_ce":
        intensity_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            softmax=False,
            sigmoid=True,
            batch=True,
        )
        if rank == 0:
            print("Use dice ce loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=args.spatial_dims, network_type="squeeze")
    loss_perceptual.to(device)

    adv_weight = 0.5
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter.
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    ohe = OHE(to_onehot=args.num_classes, dim=1)
    
    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    best_val_recon_epoch_loss = 100.0
    total_step = 0
    recons_loss = 0.0
    
    print("\nStart training autoencoder...")
    for epoch in range(max_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            
        for step, batch in enumerate(train_loader):
            progress_bar(
                step,
                len(train_loader),
                f"epoch {epoch}, \
                    Total Loss: {recons_loss if (step > 1) else 0:.4f}",
            )
            
            images = batch["label"].to(device)
            if args.num_classes > 1:
                images = ohe(images)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)

            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            if args.num_classes > 1:
                p_loss = loss_perceptual(F.softmax(reconstruction, dim=1).argmax(1, keepdim=True).float(), images.argmax(1, keepdim=True).float())
            else:
                p_loss = loss_perceptual((torch.sigmoid(reconstruction) > 0.5).float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
                tensorboard_writer.add_scalar("train_kl_loss_iter", kl_loss, total_step)
                tensorboard_writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
                    tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
                    tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

        # validation
        if (epoch) % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["label"].to(device)
                if args.num_classes > 1:
                    images = ohe(images)
                    
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = intensity_loss(
                        reconstruction.float(), images.float()
                    )
                    if args.num_classes > 1:
                        p_loss = loss_perceptual(F.softmax(reconstruction, dim=1).argmax(1, keepdim=True).float(), images.argmax(1, keepdim=True).float())
                    else:
                        p_loss = loss_perceptual((torch.sigmoid(reconstruction) > 0.5).float(), images.float())
                    recons_loss += kl_weight * KL_loss(z_mu, z_sigma) + perceptual_weight * p_loss

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save last model
                print(f"Epoch {epoch} val_loss: {val_recon_epoch_loss}")
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                mid_slice = images.shape[0] // 2
                tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
                tensorboard_writer.add_image(
                    "val_img",
                    visualize_2d_image(images[mid_slice].argmax(0) if args.num_classes > 1 else images[mid_slice, 0]).transpose([2, 1, 0]),
                    epoch,
                )
                tensorboard_writer.add_image(
                    "val_recon",
                    visualize_2d_image(F.softmax(reconstruction[mid_slice], dim=0).argmax(0) if args.num_classes > 1 else (F.sigmoid(reconstruction[mid_slice, 0]) > 0.5).float()).transpose([2, 1, 0]),
                    epoch,
                )        
            print()
        

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
