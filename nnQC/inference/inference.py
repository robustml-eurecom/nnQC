#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import os
import nibabel as nib
import argparse
import json

from medpy.metric.binary import dc
from scipy.ndimage import center_of_mass, shift

from monai.utils import set_determinism
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDIMScheduler
from monai import transforms

from nnQC.utils.utils import define_instance, corrupt_ohe_masks, create_transforms
from nnQC.models.xa import CLIPCrossAttentionGrid
from nnQC.inference.evaluation import align_center_of_mass

import warnings
warnings.filterwarnings("ignore")


def load_models(args, device):
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(args.model_dir, "autoencoder.pt"), weights_only=True))
    
    diffusion_model = define_instance(args, "diffusion_def").to(device)
    diffusion_model.load_state_dict(torch.load(os.path.join(args.model_dir, "diffusion_unet_last.pt"), weights_only=True))
    
    xa = CLIPCrossAttentionGrid(output_dim=512, grid_reduction='column_softmax').to(device)
    xa.load_state_dict(torch.load(os.path.join(args.model_dir, "xa_last.pt"), weights_only=True))
    
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), 
        torch.nn.GELU(), 
        torch.nn.Linear(32, args.diffusion_def['cross_attention_dim'])
        ).to(device)
    embed.load_state_dict(torch.load(os.path.join(args.model_dir, "embed_last.pt"), weights_only=True))
    
    scheduler = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=args.NoiseScheduler["clip_sample"],
    )
    
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.02)
    
    return autoencoder, diffusion_model, xa, embed, scheduler, inferer


def generate_pgt(autoencoder, diffusion_model, xa, embed, scheduler, inferer, processed_data, args, device):
    autoencoder.eval()
    diffusion_model.eval()
    xa.eval()
    embed.eval()
    
    images = processed_data["image"].float().to(device)
    labels = processed_data["label"].float().to(device)
    slice_ratios = processed_data["slice_label"].unsqueeze(1).float().to(device)
    
    with torch.no_grad():
        #sample_latent = autoencoder.encode_stage_2_inputs(labels[:1])
        #scale_factor = 1 / torch.std(sample_latent)
        scale_factor = torch.tensor(1.03).to(device)  # Assuming scale factor is 1 for simplicity
        inferer.scale_factor = scale_factor.item()
        scheduler.set_timesteps(5)
        
        if args.num_classes > 1:
            labels = transforms.AsDiscrete(to_onehot=args.num_classes, dim=1)(labels)
        corr_masks = corrupt_ohe_masks(
            labels, 
            corruption_prob=1.0, 
            fp_prob=0.9, 
            hole_prob=0.5, 
            erosion_prob=0.3, 
            max_operations=3
            ).argmax(1, keepdim=True) / args.num_classes
        
        slice_embeddings = embed(slice_ratios).float().to(device)
        c, _, _ = xa(images, ext_features=slice_embeddings)
        c = c.float().to(device).unsqueeze(1)
        
        latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
        noise_shape = [corr_masks.shape[0], args.latent_channels] + latent_shape
        true_noise = torch.randn(noise_shape, dtype=torch.float32).to(device)
        mask_resized = torch.nn.functional.interpolate(corr_masks.float(), size=latent_shape, mode='nearest')
        noise = torch.cat([true_noise, mask_resized], dim=1)
        
        synthetic_masks = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            conditioning=c,
        )
        
        if args.num_classes > 1:
            synth_raw = torch.argmax(synthetic_masks, dim=1, keepdim=True)
        else:
            synth_raw = torch.sigmoid(synthetic_masks) > 0.5
        
        gt_np = labels.argmax(1, keepdim=True).cpu().numpy()
        corr_np = corr_masks.cpu().numpy()
        
        slice_metrics = {}
        gt_slice_metrics = {}
        
        # Simplified processing without postprocess for now
        for slice_idx in range(len(synth_raw)):
            gt_slice = gt_np[slice_idx, 0]
            corr_slice = corr_np[slice_idx, 0]
            synth_slice = synth_raw[slice_idx, 0]  # Remove channel dimension
            #aligned_synth_slice = align_center_of_mass(synth_slice, gt_slice)
            aligned_synth_slice = synth_slice
            
            if gt_slice.any() and aligned_synth_slice.any():
                try:
                    slice_dsc = dc(aligned_synth_slice, corr_slice)
                    slice_metrics[f'slice_{slice_idx}'] = slice_dsc
                    
                    gt_slice_dsc = dc(gt_slice, corr_slice)
                    gt_slice_metrics[f'slice_{slice_idx}'] = gt_slice_dsc
                except:
                    slice_metrics[f'slice_{slice_idx}'] = 0.0
                    gt_slice_metrics[f'slice_{slice_idx}'] = 0.0

    if args.using_gt:
        return gt_np, corr_masks, synth_raw, slice_metrics, gt_slice_metrics
    else:
        return corr_masks, synth_raw, slice_metrics


def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("-e", "--environment-file", default="./config/environment.json")
    parser.add_argument("-c", "--config-file", default="./config/config_train_32g.json")
    parser.add_argument("-x", "--corrupt", action="store_true", help="Whether to corrupt the input masks")
    
    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    autoencoder, diffusion_model, xa, embed, scheduler, inferer = load_models(args, device)
    
    val_transforms = create_transforms(args, channel=args.channel, sample_axis=args.sample_axis)
    
    data_dict = {"image": args.image_path, "label": args.label_path}
    processed_data = val_transforms(data_dict)
    print("Input data processed has shape:", processed_data["image"].shape, "\n")
    
    input_mask, corr_masks, generated_masks, slice_metrics, gt_slice_metrics = generate_pgt(
        autoencoder, diffusion_model, xa, embed, scheduler, inferer, processed_data, args, device
    )
    
    # Define post-processing transforms for volume reconstruction
    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="label",
            meta_keys="pred_meta_dict",
            orig_meta_keys="label_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )
    ])
    
    volume_slices = []
    corr_volume_slices = []
    total_slices = generated_masks.shape[0]
    post = transforms.KeepLargestConnectedComponent(
            applied_labels=list(range(1, args.num_classes)) if args.num_classes > 1 else [1],
            is_onehot=False,
            connectivity=1,
            independent=False,
        )
    
    for i in range(total_slices):
        slice_data = {
            "pred": torch.tensor(generated_masks[i][None]).float(),
            "pred_meta_dict": processed_data.get("label_meta_dict", {}),
        }
        
        corr_data = {
            "pred": torch.tensor(corr_masks[i][None]).float(),
            "pred_meta_dict": processed_data.get("label_meta_dict", {}),
        }
        
        applied_labels = list(range(1, args.num_classes)) if args.num_classes > 1 else [1]
        processed_slice = post_transforms(slice_data)
        processed_corr_slice = post_transforms(corr_data)
        
        pred_np = processed_slice["pred"].cpu().numpy().squeeze()
        corr_np = processed_corr_slice["pred"].cpu().numpy().squeeze()
        
        volume_slices.append(processed_slice["pred"])
        corr_volume_slices.append(torch.tensor(corr_np).unsqueeze(0))  # Add channel dim back
        

    if volume_slices:
        reconstructed_volume = torch.stack(volume_slices, dim=-1).squeeze().cpu().numpy()
    else:
        reconstructed_volume = np.zeros_like(nib.load(args.label_path).get_fdata())
    
    corr_volume = torch.stack(corr_volume_slices, dim=-1).squeeze().cpu().numpy()

    print(f"Volume reconstruction completed. Final shape: {reconstructed_volume.shape}")
    print(f"Final unique values in volume: {np.unique(reconstructed_volume)}\n")
    
    # Save the entire volume with original metadata
    original_nii = nib.load(args.label_path)
    pgt_output_path = os.path.join(args.save_dir, args.label_path.split('/')[-1].replace('.nii.gz', '_pgt.nii.gz'))
    
    nii_img = nib.Nifti1Image(reconstructed_volume.astype(np.float32), original_nii.affine, original_nii.header)
    nib.save(nii_img, pgt_output_path)
    
    nii_corr_img = nib.Nifti1Image(corr_volume, original_nii.affine, original_nii.header)
    nib.save(nii_corr_img, os.path.join(args.save_dir, args.label_path.split('/')[-1].replace('.nii.gz', '_corr.nii.gz')))
    print(f"Generated pGT volume saved to: {pgt_output_path}")
    
    overall_dsc = np.mean([dsc for dsc in slice_metrics.values() if not np.isnan(dsc)])
    
    print(f"Overall DSC: {overall_dsc:.4f}")
    for slice_name, dsc in slice_metrics.items():
        if dsc < 0.5:
            print(f"Warning: {slice_name} has low DSC ({dsc:.4f})")
        else:
            print(f"{slice_name}: {dsc:.4f}")
    
    print()
    if args.using_gt:
        overall_gt_dsc = np.mean([dsc for dsc in gt_slice_metrics.values() if not np.isnan(dsc)])
        print(f"Overall GT DSC: {overall_gt_dsc:.4f}")
        for slice_name, dsc in gt_slice_metrics.items():
            if dsc < 0.5:
                print(f"Warning: {slice_name} has low GT DSC ({dsc:.4f})")
            else:
                print(f"{slice_name}: {dsc:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame([{"slice_name": k, "dsc": v} for k, v in slice_metrics.items()])
    metrics_df.to_csv(os.path.join(args.save_dir, args.label_path.split('/')[-1].replace('.nii.gz', '_metrics.csv')), index=False)
    
    return overall_dsc, slice_metrics


if __name__ == "__main__":
    overall_dsc, slice_metrics = run_inference()