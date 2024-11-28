import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.amp import autocast, GradScaler
from medpy.metric import binary
from scipy.ndimage import binary_erosion
from .networks import MaskLoss, ImageLoss
from .gans import LatentPerceptualLoss
import torch.nn.functional as F
from monai.losses import GeneralizedDiceLoss, PerceptualLoss, PatchAdversarialLoss
from torch.optim.lr_scheduler import ExponentialLR


def find_empty_masks(images):
    for i in range(images.size(0)):
        if images[i].sum() == 0:
            return True
    

def erode_ohe_mask(mask, target_pixels=8):
    mask_np = mask.cpu().numpy()
    eroded_mask_np = np.zeros_like(mask_np)

    for i in range(mask_np.shape[0]):
        class_mask = mask_np[i, :, :]

        while class_mask.sum() > target_pixels:
            class_mask = binary_erosion(class_mask)

        eroded_mask_np[i, :, :] = class_mask
        
    eroded_mask = torch.from_numpy(eroded_mask_np).to(mask.device)
    return eroded_mask


def create_holes(mask, min_radius, max_radius, max_holes):
    mask_np = mask.cpu().numpy()
    modified_mask = mask_np.copy()
    non_zero_indices = np.transpose(np.nonzero(mask_np))

    # Randomly select a subset of the non-zero voxels
    selected_indices = random.sample(list(non_zero_indices), min(max_holes, len(non_zero_indices)))

    for idx in selected_indices:
        radius = random.randint(min_radius, max_radius)

        # Create a spherical mask
        sphere_mask = np.zeros_like(mask_np)
        y,x,z = np.ogrid[-idx[0]:mask_np.shape[0]-idx[0], -idx[1]:mask_np.shape[1]-idx[1], -idx[2]:mask_np.shape[2]-idx[2]]
        mask_sphere = x*x + y*y + z*z <= radius*radius
        sphere_mask[mask_sphere] = 1

        # Create a hole at the current voxel
        modified_mask = np.where(sphere_mask, np.min(mask_np), modified_mask)  # assuming the background value is the minimum value

    modified_mask = torch.from_numpy(modified_mask).to(mask.device)
    return modified_mask


def bce_loss(prediction, target):
    return nn.BCEWithLogitsLoss()(prediction, target)

def latent_consistency_loss(gt_latent, reconstructed_latent):
    return nn.MSELoss()(gt_latent, reconstructed_latent)

def cycle_consistency_loss(ae, gt_latent, generated_masks):
    reconstructed_latent = ae.encode(generated_masks)
    return nn.MSELoss()(gt_latent, reconstructed_latent)

def recon_loss(prediction, target, epoch, functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss):
    return MaskLoss(functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss)(prediction, target, epoch)
    

class AETrainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        loss_function,
        keys,
        device, 
        mode
    ):
        self.model = model
        self.keys = keys
        self.optimizer = optimizer
        self.device = device
        self.mode = mode  # 'image' or 'mask'
        self.loss_function = loss_function
        self.metrics = Metrics()
        #self.scaler = GradScaler()

    def train(self, train_loader, val_loader, epochs, val_interval, ckpt_folder, logs_folder):
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(epoch_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}')

            # Validation
            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
                val_losses.append(val_loss)

                print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')
                self._print_metrics(val_metrics)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f'Saving best model at epoch {epoch+1}')
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        },
                        f'{ckpt_folder}/best_model.pth'
                    )

            if epoch % 5 == 0:
                self._visualize_results(val_loader, epoch, logs_folder)

            print('--------------------------------------------------')

        torch.save(
            self.model.state_dict(),
            f'{ckpt_folder}/final_model.pth'
        )
        return train_losses, val_losses

    def _train_epoch(self, train_loader, epoch):
        total_loss = 0
        num_slices = 0
        progress_bar = tqdm(enumerate(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, batch in progress_bar:
            self.optimizer.zero_grad(set_to_none=True)
            
            #with autocast(enabled=True):
            input_data = batch['img'].to(self.device) if 'img' in self.mode else batch['seg'].to(self.device)
            reconstruction = self.model(input_data)
            loss = self.loss_function(reconstruction, input_data, epoch)

            num_slices += input_data.size(0)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        return total_loss / num_slices

    def _validate_epoch(self, val_loader, epoch):
        total_val_loss = 0
        num_slices = 0
        val_metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                input_data = batch['img'].to(self.device) if 'img' in self.mode else batch['seg'].to(self.device)
                reconstruction = self.model(input_data)
                val_loss = self.loss_function(reconstruction, input_data, epoch)
                total_val_loss += val_loss.item()
                num_slices += input_data.size(0)
                if 'mask' in self.mode:
                    gt = np.argmax(input_data.cpu().numpy(), axis=1)
                    pred = np.argmax(reconstruction.cpu().numpy(), axis=1)
                    batch_metrics = self.metrics(pred, gt, self.keys)
                    for k, v in batch_metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = []
                        val_metrics[k].append(v)

        avg_val_loss = total_val_loss / num_slices
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

        return avg_val_loss, avg_val_metrics

    def _print_metrics(self, metrics):
        metrics_str = "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(metrics_str)

    def _visualize_results(self, val_loader, epoch, logs_folder):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            input_data = batch['img'].to(self.device) if 'img' in self.mode else batch['seg'].to(self.device)
            reconstruction = self.model(input_data)

            inp_img = input_data[0, 0].cpu().numpy() if 'img' in self.mode else input_data[0].argmax(0).cpu().numpy()
            out_img = reconstruction[0, 0].cpu().numpy() if 'img' in self.mode else reconstruction[0].argmax(0).cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(inp_img, cmap="gray")
            ax[0].set_title(f"Original {self.mode}")
            ax[1].imshow(out_img, cmap="gray")
            ax[1].set_title(f"Reconstructed {self.mode}")
            plt.show()
            plt.savefig(f'{logs_folder}/reconstruction_epoch_{epoch}.png')
            plt.close()


class GANTrainer:
    def __init__(
        self, 
        generator, 
        discriminator, 
        g_optimizer, 
        d_optimizer, 
        device, 
        feature_extractor, 
        ae, 
        loss_functions,
        lambda_gan=0.5
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.img_ae = feature_extractor
        self.mask_ae = ae
        self.lambda_gan = lambda_gan
        self.loss_functions = loss_functions
        self.metrics = Metrics()
        #self.scaler = GradScaler()

    def train(
        self, 
        train_loader, 
        val_loader, 
        num_epochs, 
        val_interval,
        ckpt_folder,
        logs_folder,
    ):
        
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            train_g_loss, train_d_loss = self._train_epoch(train_loader, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train D Loss: {train_d_loss:.4f}, Train G Loss: {train_g_loss:.4f}')

            self.generator.eval()
            self.discriminator.eval()
            val_g_loss, val_d_loss = self._validate_epoch(val_loader, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation D Loss: {val_d_loss:.4f}, Validation G Loss: {val_g_loss:.4f}')

            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                print(f'Saving best model at epoch {epoch+1}')
                torch.save({
                    'G': self.generator.state_dict(),
                    'D': self.discriminator.state_dict(),
                    'G_optim': self.g_optimizer.state_dict(),
                    'D_optim': self.d_optimizer.state_dict()
                }, f'{ckpt_folder}/best_model.pth')

            if (epoch + 1) % val_interval == 0:
                self._visualize_results(val_loader, logs_folder, epoch)

            print('--------------------------------------------------')

    def _train_epoch(self, train_loader, epoch):
        total_g_loss = 0
        total_d_loss = 0
        num_slices = 0
        progress_bar = tqdm(enumerate(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for step, batch in progress_bar:
            prob = np.random.rand()
            target_pix_choice = np.random.choice([8, 16, 32])
            scans, gt_masks = batch['img'], batch['seg']
            scans, gt_masks = scans.to(self.device), gt_masks.to(self.device)
            corrupted_masks = torch.stack([erode_ohe_mask(mask, target_pix_choice) for mask in gt_masks]) if prob < 0.5 else gt_masks

            batch_size = scans.size(0)
            num_slices += batch_size
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            # ---- Train Discriminator ----
            self.discriminator.zero_grad()
            
            real_outputs = self.discriminator(gt_masks)
            d_loss_real = bce_loss(real_outputs, real_labels)
            
            features = self.img_ae.encode(scans).to(self.device)
            embeddings = self.mask_ae.encode(corrupted_masks).to(self.device)

            generated_masks = self.generator(features, embeddings)
            fake_outputs = self.discriminator(generated_masks.detach())
            d_loss_fake = bce_loss(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # ---- Train Generator ----
            self.generator.zero_grad()

            # GAN loss
            fake_outputs = self.discriminator(generated_masks)
            g_loss_gan = self.lambda_gan * bce_loss(fake_outputs, real_labels) \
                + (1 - self.lambda_gan) * recon_loss(generated_masks, gt_masks, epoch, **self.loss_functions)

            # Latent perceptual loss
            gt_latent = self.mask_ae.encode(gt_masks)
            latent_perceptual_loss = LatentPerceptualLoss()(gt_latent, self.mask_ae.encode(generated_masks))

            ## Cycle consistency loss
            #cycle_loss_value = cycle_consistency_loss(self.mask_ae, gt_latent, generated_masks)

            # Total generator loss
            g_loss = g_loss_gan + 2 * latent_perceptual_loss #+ cycle_loss_value
            g_loss.backward()
            self.g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            progress_bar.set_postfix({"G loss": total_g_loss / (step + 1), "D loss": total_d_loss / (step + 1)})
            
        return total_g_loss / (step+1), total_d_loss / (step+1)

    def _validate_epoch(self, val_loader, epoch):
        total_g_loss = 0
        total_d_loss = 0
        num_slices = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                prob = np.random.rand()
                target_pix_choice = np.random.choice([8, 16, 32])
                scans, gt_masks = batch['img'], batch['seg']
                scans, gt_masks = scans.to(self.device), gt_masks.to(self.device)
                corrupted_masks = torch.stack([erode_ohe_mask(mask, target_pix_choice) for mask in gt_masks]) if prob < 0.5 else gt_masks

                batch_size = scans.size(0)
                num_slices += batch_size

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # ---- Discriminator ----
                real_outputs = self.discriminator(gt_masks)
                d_loss_real = bce_loss(real_outputs, real_labels)

                features = self.img_ae.encode(scans).to(self.device)
                embeddings = self.mask_ae.encode(corrupted_masks).to(self.device)

                generated_masks = self.generator(features, embeddings)
                fake_outputs = self.discriminator(generated_masks.detach())
                d_loss_fake = bce_loss(fake_outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                # ---- Generator ----
                fake_outputs = self.discriminator(generated_masks)
                g_loss_gan = self.lambda_gan * bce_loss(fake_outputs, real_labels) \
                    + (1 - self.lambda_gan) * recon_loss(generated_masks, gt_masks, epoch, **self.loss_functions)

                gt_latent = self.mask_ae.encode(gt_masks)
                latent_perceptual_loss = LatentPerceptualLoss()(gt_latent, self.mask_ae.encode(generated_masks))

                g_loss = g_loss_gan + 2 * latent_perceptual_loss #+ cycle_loss_value

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

        return total_g_loss / (i+1), total_d_loss / (i+1)

    def _visualize_results(self, val_loader, logs_folder, epoch):
        self.generator.eval()
        with torch.no_grad():
            target_pix_choice = np.random.choice([8, 16, 32])
            batch = next(iter(val_loader))
            scans, gt_masks = batch['img'], batch['seg']
            scans, gt_masks = scans.to(self.device), gt_masks.to(self.device)
            corrupted_masks = torch.stack([erode_ohe_mask(mask, target_pix_choice) for mask in gt_masks])

            features = self.img_ae.encode(scans).to(self.device)
            embeddings = self.mask_ae.encode(corrupted_masks).to(self.device)
            generated_masks = self.generator(features, embeddings)

            fig, ax = plt.subplots(1, 4, figsize=(10, 5))
            ax[0].imshow(gt_masks[0].cpu().numpy().argmax(0), cmap="gray")
            ax[0].set_title("Ground truth")
            ax[1].imshow(scans[0, 0].cpu().numpy(), cmap="gray")
            ax[1].set_title("Input image")
            ax[2].imshow(corrupted_masks[0].cpu().numpy().argmax(0), cmap="gray")
            ax[2].set_title("Input corr. mask")
            ax[3].imshow(generated_masks[0].cpu().numpy().argmax(0), cmap="gray")
            ax[3].set_title("Generated")
            
            plt.savefig(f'{logs_folder}/generated_masks_epoch_{epoch}.png')
            plt.close()


class AutoencoderKLTrainer:
    def __init__(
        self,
        autoencoderkl,
        discriminator,
        device,
    ):
        self.autoencoderkl = autoencoderkl.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.perceptual_loss = self._init_perceptual_loss()
        self.dice_loss = GeneralizedDiceLoss(include_background=False)
        self.adv_loss = self._init_adversarial_loss()

        self.optimizer_g = torch.optim.Adam(self.autoencoderkl.parameters(), lr=5e-5)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        self._init_tracking_variables()

    def _init_perceptual_loss(self):
        perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg")
        return perceptual_loss.to(self.device)

    def _init_adversarial_loss(self):
        return PatchAdversarialLoss(criterion="least_squares")

    def _init_folders(self):
        if not os.path.exists(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)

    def _init_tracking_variables(self):
        self.epoch_recon_losses = []
        self.epoch_gen_losses = []
        self.epoch_disc_losses = []
        self.val_recon_losses = []
        self.intermediary_images = []
        self.num_example_images = 5
        self.min_val_loss = 1e6

    def train(
        self,
        train_loader,
        val_loader,
        ckpt_folder='checkpoints',
        epochs=30,
        val_interval=5,
        kl_weight=1e-6,
        perceptual_weight=0.001,
        adv_weight=0.01,
        autoencoder_warm_up_n_epochs=5,
        logs_folder='logs'
        ):
               
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.ckpt_folder = ckpt_folder
        self.logs_folder = logs_folder
        self.n_epochs = epochs
        self.val_interval = val_interval
        self.autoencoder_warm_up_n_epochs = autoencoder_warm_up_n_epochs
        
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        
        self._init_folders()
        
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            
            if (epoch + 1) % self.val_interval == 0:
                self._validate_epoch(epoch)

        self._save_final_model(epoch)
        self._cleanup()

    def _train_epoch(self, epoch):
        self.autoencoderkl.train()
        self.discriminator.train()
        epoch_loss = gen_epoch_loss = disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            images = batch["seg"].to(self.device)

            if find_empty_masks(images):
                continue
            
            loss_g, generator_loss, reconstruction = self._train_generator(images, epoch)
            epoch_loss += loss_g.item()

            if epoch > self.autoencoder_warm_up_n_epochs:
                loss_d = self._train_discriminator(images, reconstruction)
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += loss_d.item()

            progress_bar.set_postfix({
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            })

        self._update_epoch_losses(epoch_loss, gen_epoch_loss, disc_epoch_loss, step)

    def _train_generator(self, images, epoch):
        self.optimizer_g.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = self.autoencoderkl(images)
            recons_loss = self.dice_loss(reconstruction.float(), images.float())
            kl_loss = self._compute_kl_loss(z_mu, z_sigma)
            
            reconstruction_3ch = reconstruction.argmax(1, keepdim=True).repeat(1, 3, 1, 1).float()
            images_3ch = images.argmax(1, keepdim=True).repeat(1, 3, 1, 1).float()           
            p_loss = self.perceptual_loss(reconstruction_3ch.float(), images_3ch.float())
            
            loss_g = recons_loss + (self.kl_weight * kl_loss) + (self.perceptual_weight * p_loss)
            generator_loss = 0
            if epoch > self.autoencoder_warm_up_n_epochs:
                generator_loss = self._compute_generator_loss(reconstruction)
                loss_g += self.adv_weight * generator_loss

        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return loss_g, generator_loss, reconstruction

    def _train_discriminator(self, images, reconstruction): 
        with autocast(enabled=True):
            self.optimizer_d.zero_grad(set_to_none=True)
            
            loss_d_fake = self._compute_discriminator_loss(reconstruction, False)
            loss_d_real = self._compute_discriminator_loss(images, True)
            loss_d = self.adv_weight * ((loss_d_fake + loss_d_real) * 0.5)

        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        return loss_d

    def _compute_kl_loss(self, z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)+ 1e-6) - 1, dim=[1, 2, 3])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    def _compute_generator_loss(self, reconstruction):
        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
        return self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

    def _compute_discriminator_loss(self, images, target_is_real):
        logits = self.discriminator(images.contiguous().detach())[-1]
        return self.adv_loss(logits, target_is_real=target_is_real, for_discriminator=True)

    def _update_epoch_losses(self, epoch_loss, gen_epoch_loss, disc_epoch_loss, step):
        self.epoch_recon_losses.append(epoch_loss / (step + 1))
        self.epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        self.epoch_disc_losses.append(disc_epoch_loss / (step + 1))

    def _validate_epoch(self, epoch):
        self.autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch["seg"].to(self.device)
                
                with autocast(enabled=True):
                    reconstruction, _, _ = self.autoencoderkl(images)
                    if val_step == 1:
                        self.intermediary_images.append(reconstruction[:self.num_example_images].argmax(1))
                    
                    recons_loss = self.dice_loss(images.float(), reconstruction.float())
                
                val_loss += recons_loss.item()

        self._visualize_reconstructions(epoch)
        self._update_validation_loss(val_loss, val_step, epoch)

    def _visualize_reconstructions(self, epoch):
        fig, axes = plt.subplots(nrows=1, ncols=self.num_example_images, sharex=True, figsize=(10, 20))
        for i in range(self.num_example_images):
            axes[i].imshow(self.intermediary_images[-1][i].cpu().numpy(), cmap="gray")
            axes[i].set_title(f"Reconstruction {i+1}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(f'{self.logs_folder}/reconstruction_epoch_{epoch}.png')
        plt.close()

    def _update_validation_loss(self, val_loss, val_step, epoch):
        val_loss /= val_step
        self.val_recon_losses.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
        
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self._save_best_model(epoch)

    def _save_best_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'autoencoderkl_state_dict': self.autoencoderkl.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'epoch_recon_losses': self.epoch_recon_losses,
            'epoch_gen_losses': self.epoch_gen_losses,
            'epoch_disc_losses': self.epoch_disc_losses,
            'val_recon_losses': self.val_recon_losses,
            'intermediary_images': self.intermediary_images
        }, os.path.join(self.ckpt_folder, 'best.pth'))

    def _save_final_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'autoencoderkl_state_dict': self.autoencoderkl.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'epoch_recon_losses': self.epoch_recon_losses,
            'epoch_gen_losses': self.epoch_gen_losses,
            'epoch_disc_losses': self.epoch_disc_losses,
            'val_recon_losses': self.val_recon_losses,
            'intermediary_images': self.intermediary_images
        }, os.path.join(self.ckpt_folder, 'final.pth'))

    def _cleanup(self):
        del self.discriminator
        del self.perceptual_loss
        torch.cuda.empty_cache()
        

class LDMTrainer:
    def __init__(self, unet, spatial_ae, feat_extractor, mask_ae, inferer, scheduler, train_loader, val_loader, device):
        self.unet = unet.to(device)
        self.spatial_ae = spatial_ae
        self.feat_extractor = feat_extractor
        self.mask_ae = mask_ae
        self.inferer = inferer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = GradScaler()
        self.epoch_losses = []
        self.val_losses = []
        

    def train(self, ckpt_folder, logs_folder, epochs=200, val_interval=5, lr_start=1e-4, lr_end=1e-6, gamma=None):
        self.n_epochs = epochs
        self.val_interval = val_interval
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr_start)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        for epoch in range(self.n_epochs):
            self.unet.train()
            self.spatial_ae.eval()
            self.feat_extractor.eval()
            self.mask_ae.eval()
            
            epoch_loss = self._train_epoch(epoch)

            print(f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {epoch_loss:.4f}')

            if (epoch) % self.val_interval == 0:
                self.unet.eval()
                val_loss = self._validate_epoch(epoch)

                print(f'Epoch [{epoch+1}/{self.n_epochs}], Validation Loss: {val_loss:.4f}')

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'unet_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch_losses': self.epoch_losses,
                    'val_losses': self.val_losses
                }, os.path.join(ckpt_folder, 'best.pth'))

                # Sampling image during training
                self._sample_image(epoch, logs_folder)

            print('--------------------------------------------------')
        
        torch.save({
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_losses': self.epoch_losses,
            'val_losses': self.val_losses
        }, os.path.join(ckpt_folder, 'final.pth'))

    def _train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(enumerate(self.train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, batch in progress_bar:
            images = batch["img"].to(self.device)
            segmentations = batch["seg"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            prob = np.random.rand()
            
            if find_empty_masks(segmentations):
                continue
            
            with autocast(enabled=True):
                z_mu, z_sigma = self.spatial_ae.autoencoderkl.encode(segmentations)
                z = self.spatial_ae.autoencoderkl.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(self.device)
                condition = self.feat_extractor.encode(images).to(self.device)
                #condition = F.normalize(condition, p=2, dim=-1)
                condition = condition.view(segmentations.shape[0], -1).unsqueeze(1)
                
                if self.mask_ae is not None:
                    holes_segmentations = torch.stack([create_holes(mask, 15, 30, 10) for mask in segmentations]) if prob < 0.5 else segmentations
                    mask_condition = self.mask_ae.encode(holes_segmentations).to(self.device)
                    #mask_condition = F.normalize(mask_condition, p=2, dim=-1)
                    mask_condition = mask_condition.view(segmentations.shape[0], -1).unsqueeze(1)
                    condition = torch.cat([condition, mask_condition], dim=-1)
                
                timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                noise_pred = self.inferer(
                    inputs=segmentations,
                    diffusion_model=self.unet,
                    noise=noise,
                    timesteps=timesteps,
                    autoencoder_model=self.spatial_ae,
                    condition=condition,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": total_loss / (step+1)})
        self.lr_scheduler.step()
        self.epoch_losses.append(total_loss / (step+1))
        return total_loss / (step+1)

    def _validate_epoch(self, epoch):
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch["img"].to(self.device)
                segmentations = batch["seg"].to(self.device)
                prob = np.random.rand()
                
                with autocast(enabled=True):
                    z_mu, z_sigma = self.spatial_ae.autoencoderkl.encode(segmentations)
                    z = self.spatial_ae.autoencoderkl.sampling(z_mu, z_sigma)
                    noise = torch.randn_like(z).to(self.device)
                    condition = self.feat_extractor.encode(images).to(self.device).view(segmentations.shape[0], -1).unsqueeze(1)
                    if self.mask_ae is not None:
                        holes_segmentations = torch.stack([create_holes(mask, 15, 30, 10) for mask in segmentations]) if prob < 0.5 else segmentations
                        mask_condition = self.mask_ae.encode(holes_segmentations).to(self.device).view(segmentations.shape[0], -1).unsqueeze(1)
                        condition = torch.cat([condition, mask_condition], dim=-1)
                    
                    timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    noise_pred = self.inferer(
                        inputs=segmentations,
                        diffusion_model=self.unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=self.spatial_ae,
                        condition=condition,
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()

        val_loss /= (val_step+1)
        self.val_losses.append(val_loss)
        return val_loss

    def _sample_image(self, epoch, logs_folder):
        segmentations = next(iter(self.val_loader))["seg"].to(self.device)
        images = next(iter(self.val_loader))["img"].to(self.device)

        z = torch.rand_like(
            self.spatial_ae.encode_stage_2_inputs(segmentations[0].unsqueeze(0))
            ).to(self.device)
        condition = self.feat_extractor.encode(images[0].unsqueeze(0)).to(self.device).view(1, -1).unsqueeze(1)
        if self.mask_ae is not None:
            holes_segmentations = create_holes(segmentations[0], 15, 30, 10)
            mask_condition = self.mask_ae.encode(holes_segmentations.unsqueeze(0)).to(self.device).view(1, -1).unsqueeze(1)
            condition = torch.cat([condition, mask_condition], dim=-1)
        self.scheduler.set_timesteps(num_inference_steps=1000)

        with autocast(enabled=True):
            decoded = self.inferer.sample(
                input_noise=z, diffusion_model=self.unet, scheduler=self.scheduler, 
                autoencoder_model=self.spatial_ae, conditioning=condition
            )
            reconstructed, _, _ = self.spatial_ae(segmentations[0].unsqueeze(0))

        fig, ax = plt.subplots(1, 4, figsize=(12, 6))
        ax[0].imshow(images[0, 0].cpu().numpy(), cmap="gray")
        ax[0].set_title("Original image")
        ax[0].axis("off")
        ax[1].imshow(segmentations[0].argmax(0).cpu().numpy(), cmap="gray")
        ax[1].set_title("Ground truth")
        ax[1].axis("off")
        ax[2].imshow(reconstructed[0].argmax(0).cpu().numpy(), cmap="gray")
        ax[2].set_title("Reconstructed")
        ax[2].axis("off")
        ax[3].imshow(decoded[0].argmax(0).cpu().numpy(), cmap="gray")
        ax[3].set_title("Sampled")
        ax[3].axis("off")
        plt.savefig(f'{logs_folder}/sampled_image_epoch_{epoch}.png')

            
class Metrics:
    def __init__(self):
        self.DC = DC()
        self.HD = HD()

    def __call__(self, prediction, target, keys):
        metrics = {}
        for c, key in enumerate(keys):
            ref = np.copy(target)
            pred = np.copy(prediction)

            ref = np.where(ref != c, 0, 1)
            pred = np.where(pred != c, 0, 1)

            metrics[key + "_dc"] = self.DC(pred, ref)
            metrics[key + "_hd"] = self.HD(pred, ref)
        return metrics

class DC:
    def __call__(self, prediction, target):
        try:
            return binary.dc(prediction, target)
        except Exception:
            return 0

class HD:
    def __call__(self, prediction, target):
        try:
            return binary.hd(prediction, target)
        except Exception:
            return np.nan