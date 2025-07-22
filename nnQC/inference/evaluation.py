import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse
import json
import nibabel as nib
from datetime import datetime
import random

from medpy.metric.binary import dc, hd95
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass, shift

from monai.utils import set_determinism
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.data.utils import first
from monai.transforms import KeepLargestConnectedComponent
from monai.transforms import AsDiscrete as OHE

from nnQC.utils import define_instance
from nnQC.models.xa import CLIPCrossAttentionGrid

def ohe(mask, num_classes: int = 2):
    return OHE(to_onehot=num_classes, dim=1)(mask)

def align_center_of_mass(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Align the center of mass of predicted mask to ground truth mask.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Aligned predicted mask
    """
    if not pred_mask.any() or not gt_mask.any():
        return pred_mask
    
    try:
        # Calculate centers of mass
        pred_com = center_of_mass(pred_mask.astype(float))
        gt_com = center_of_mass(gt_mask.astype(float))
        
        # Calculate shift needed
        shift_vector = [gt_com[i] - pred_com[i] for i in range(len(pred_com))]
        
        # Apply shift
        aligned_mask = shift(pred_mask.astype(float), shift_vector, order=0, mode='constant', cval=0)
        
        #return (aligned_mask > 0.5).astype(bool)
        return aligned_mask
    except:
        return pred_mask


def visualize_random_samples(batch_data: Dict, batch_idx: int, output_dir: str, num_samples: int = 3):
    """
    Visualize random samples from the batch including scan, gt, corrupted mask, and prediction.
    
    Args:
        batch_data: Dictionary containing 'scans', 'gt_masks', 'corr_masks', 'pred_masks'
        batch_idx: Batch index for naming
        output_dir: Output directory for saving plots
        num_samples: Number of random samples to visualize
    """
    scans = batch_data['scans']
    gt_masks = batch_data['gt_masks']
    corr_masks = batch_data['corr_masks']
    pred_masks = batch_data['pred_masks']
    
    batch_size = scans.shape[0]
    sample_indices = random.sample(range(batch_size), min(num_samples, batch_size))
    
    for i, sample_idx in enumerate(sample_indices):
        # Get data for this sample
        scan = scans[sample_idx]
        gt = gt_masks[sample_idx]
        corr = corr_masks[sample_idx]
        pred = pred_masks[sample_idx]
        
        # Handle 3D data by taking middle slice
        if len(scan.shape) == 3:
            mid_slice = scan.shape[0] // 2
            scan_slice = scan[mid_slice]
            gt_slice = gt[mid_slice]
            corr_slice = corr[mid_slice]
            pred_slice = pred[mid_slice]
        else:
            scan_slice = scan
            gt_slice = gt
            corr_slice = corr
            pred_slice = pred
        
        pred_slice = pred_slice.argmax(0) if pred.shape[0] > 1 else pred_slice[0]
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original scan
        axes[0].imshow(scan_slice, cmap='gray')
        axes[0].set_title('Scan')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(scan_slice, cmap='gray', alpha=0.7)
        axes[1].imshow(gt_slice, cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Corrupted mask
        axes[2].imshow(scan_slice, cmap='gray', alpha=0.7)
        axes[2].imshow(corr_slice, cmap='Blues', alpha=0.5)
        axes[2].set_title('Corrupted Mask')
        axes[2].axis('off')
        
        # Predicted mask
        axes[3].imshow(scan_slice, cmap='gray', alpha=0.7)
        axes[3].imshow(pred_slice, cmap='Greens', alpha=0.5)
        axes[3].set_title('Predicted Mask')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"batch_{batch_idx}_sample_{sample_idx}_visualization.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved visualization: {filename}")


def compute_metrics_for_validation(
    model_components: Dict[str, Any],
    val_loader,
    args,
    output_figure_dir: str = "output/figures",
    output_segmentation_dir: str = "output/segmentations",
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Compute DSC and HD95 metrics for all subjects in validation set.
    Process batches efficiently while computing slice-wise metrics.
    Run 5 different corruption variations per batch to emulate 5 different segmentation models.
    
    Args:
        model_components: Dictionary containing 'autoencoder', 'diffusion_model', 
                         'xa', 'scheduler', 'inferer'
        val_loader: Validation data loader
        args: Configuration arguments
        output_csv_path: Path to save CSV file (optional)
    
    Returns:
        pandas.DataFrame: Results with columns [batch_id, sample_idx, slice, corruption_id,
                         pseudo_dsc, pseudo_hd95, real_dsc, real_hd95]
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Unpack model components
    autoencoder = model_components['autoencoder']
    diffusion_model = model_components['diffusion_model']
    xa = model_components['xa']
    embed = model_components['embed']
    scheduler = model_components['scheduler']
    scheduler.set_timesteps(10)
    inferer = model_components['inferer']
    
    applied_labels = list(range(1, args.num_classes)) if args.num_classes > 1 else [1]
    postprocess = KeepLargestConnectedComponent(
        applied_labels=applied_labels, 
        is_onehot=False,
        connectivity=1,
        independent=False,
    )
    
    # Set models to evaluation mode
    autoencoder.eval()
    diffusion_model.eval()
    xa.eval()
    embed.eval()
    
    results = []
    
    # Define noise shape for generation
    latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
    noise_shape = [args.diffusion_train["batch_size"], args.latent_channels] + latent_shape
    
    # Define 5 different corruption configurations to emulate different segmentation models
    corruption_configs = [
        {
            'id': 'model_1_conservative',
            'corruption_prob': 1.,
            'fp_prob': 0.1,
            'hole_prob': 0.3,
            'erosion_prob': 0.2
        },
        {
            'id': 'model_2_moderate', 
            'corruption_prob': 1.,
            'fp_prob': 0.3,
            'hole_prob': 0.5,
            'erosion_prob': 0.4
        },
        {
            'id': 'model_3_aggressive',
            'corruption_prob': 1.,
            'fp_prob': 0.5,
            'hole_prob': 0.7,
            'erosion_prob': 0.6
        },
        {
            'id': 'model_4_high_fp',
            'corruption_prob': 1.,
            'fp_prob': 0.8,
            'hole_prob': 0.4,
            'erosion_prob': 0.3
        },
        {
            'id': 'model_5_high_holes',
            'corruption_prob': 1.,
            'fp_prob': 0.8,
            'hole_prob': 0.9,
            'erosion_prob': 0.7
        }
    ]
    
    logging.info(f"\nStarting evaluation on {len(val_loader)} batches with 5 corruption variations each")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
              
            gt_masks = batch['label'].to(device) 
            if args.num_classes > 1:
                gt_masks = ohe(gt_masks, args.num_classes)
            scans = batch["image"].to(device).float()  
            slice_ratios = batch["slice_label"].unsqueeze(1).to(device) 
            batch_size = gt_masks.shape[0]
            
            # Process each corruption configuration (5 different "models")
            for corruption_idx, config in enumerate(corruption_configs):
                
                from utils import corrupt_ohe_masks
                corr_masks = corrupt_ohe_masks(
                    gt_masks, 
                    corruption_prob=config['corruption_prob'], 
                    fp_prob=config['fp_prob'], 
                    hole_prob=config['hole_prob'],
                    erosion_prob=config['erosion_prob'],
                    max_operations=3
                )
                if args.num_classes > 1:
                    corr_masks = corr_masks.argmax(1, keepdim=True) / args.num_classes
                
                slice_embeddings = embed(slice_ratios).float().to(device)
                c, _, _ = xa(scans, ext_features=slice_embeddings)
                c = c.float().to(device).unsqueeze(1)
                
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
                    verbose=False
                )                
                act = torch.nn.Sigmoid() if args.num_classes == 1 else torch.nn.Softmax(dim=1)
                synth_raw = act(synthetic_masks).cpu().numpy()  
                synth_raw = synth_raw > 0.5  
                #print(f"Generated masks shape: {synth_raw.shape}, dtype: {synth_raw.dtype}\n")
                
                gt_np = gt_masks.cpu().numpy()  
                corr_np = corr_masks.cpu().numpy()
                synth_np = synth_raw

                aligned_synth_np = np.zeros_like(synth_np)
                viz_data = {
                    'scans': scans.cpu().numpy()[:, 0],  
                    'gt_masks': gt_np.argmax(1) if args.num_classes > 1 else gt_np[:, 0],
                    'corr_masks': corr_np[:, 0],
                    'pred_masks': np.zeros_like(gt_np)
                }
                
                for sample_idx in range(batch_size):
                    gt_sample = gt_np[sample_idx] if args.num_classes > 1 else gt_np[sample_idx, 0]
                    corr_sample = corr_np[sample_idx] if args.num_classes > 1 else corr_np[sample_idx, 0]
                    synth_sample = synth_np[sample_idx] if args.num_classes > 1 else synth_np[sample_idx, 0]
                    
                    synth_sample = postprocess(synth_sample)
                    #synth_sample = synth_sample.argmax(0) if args.num_classes > 1 else synth_sample
                    #aligned_synth_sample = align_center_of_mass(synth_sample, gt_sample)
                    aligned_synth_sample = synth_sample
                    
                    aligned_synth_np[sample_idx] = aligned_synth_sample
                    viz_data['pred_masks'][sample_idx] = aligned_synth_sample#.argmax(0) if args.num_classes > 1 else aligned_synth_sample[0]
                    
                    if len(gt_sample.shape) == 3 and args.num_classes == 1: 
                        #print("aaaa")
                        for slice_idx in range(gt_sample.shape[0]):
                            gt_slice = gt_sample[slice_idx]
                            corr_slice = corr_sample[slice_idx]
                            synth_slice = aligned_synth_sample[slice_idx]
                            
                            if not (gt_slice.any() or corr_slice.any() or synth_slice.any()):
                                continue
                            
                            metrics = compute_slice_metrics(
                                gt_slice, corr_slice, synth_slice, 
                                batch_idx, sample_idx, slice_idx, config['id']
                            )
                            if metrics:
                                results.append(metrics)
                    
                    else:  
                        metrics = compute_slice_metrics(
                            gt_sample, corr_sample, aligned_synth_sample, 
                            batch_idx, sample_idx, 0, config['id']
                        )
                        if metrics:
                            results.append(metrics)
                            
                # Visualize samples for first corruption only to avoid too many images
                if corruption_idx == 0 and batch_idx % 1 == 0:
                    visualize_random_samples(viz_data, batch_idx, output_figure_dir, num_samples=5)
                    
                #save nii.gz of viz_data in output_segmentation_dir, in particular pred_masks and gt_masks and corr_masks
                #np.save(os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_pred_masks.npy"), viz_data['pred_masks'])
                #np.save(os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_gt_masks.npy"), viz_data['gt_masks'])
                #np.save(os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_corr_{corruption_idx}_corr_masks.npy"), viz_data['corr_masks'])
                
                pred_nii = nib.Nifti1Image(viz_data['pred_masks'], np.eye(4), dtype=np.int8)
                gt_nii = nib.Nifti1Image(viz_data['gt_masks'], np.eye(4), dtype=np.int8)
                corr_nii = nib.Nifti1Image(viz_data['corr_masks'], np.eye(4), dtype=np.int8)
                
                nib.save(pred_nii, os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_pred_masks.nii.gz"))
                nib.save(gt_nii, os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_gt_masks.nii.gz"))
                nib.save(corr_nii, os.path.join(output_segmentation_dir, f"batch_{batch_idx}_sample_{sample_idx}_corr_{config['id']}_corr_masks.nii.gz"))
                
                logging.info(f"Processed batch {batch_idx + 1}/{len(val_loader)}, "
                           f"corruption {corruption_idx + 1}/5 ({config['id']}) with {batch_size} samples")

    
    # Create DataFrame with corruption_id column
    df = pd.DataFrame(results, columns=[
        'batch_id', 'sample_idx', 'slice', 'corruption_id', 'pseudo_dsc', 'pseudo_hd95', 'real_dsc', 'real_hd95'
    ])
    
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Results saved to {output_csv_path}")
    
    return df


def compute_slice_metrics(gt_mask: np.ndarray, corr_mask: np.ndarray, 
                         synth_mask: np.ndarray, batch_id: int, 
                         sample_idx: int, slice_idx: int, corruption_id: str) -> Dict[str, Any]:
    """
    Compute DSC and HD95 metrics for a single slice.
    
    Args:
        gt_mask: Ground truth binary mask
        corr_mask: Corrupted binary mask  
        synth_mask: Generated/synthetic binary mask
        batch_id: Batch identifier
        sample_idx: Sample index within batch
        slice_idx: Slice index
        corruption_id: Identifier for the corruption configuration
    
    Returns:
        Dictionary with computed metrics or None if computation fails
    """
    if synth_mask.ndim == 3 and synth_mask.shape[0] > 1:
        synth_mask = np.argmax(synth_mask[1:], axis=0).astype(bool)
    else:
        synth_mask = synth_mask.astype(bool)

    if corr_mask.ndim == 3 and corr_mask.shape[0] > 1:
        corr_mask = np.argmax(corr_mask[1:], axis=0).astype(bool)
    else:
        corr_mask = corr_mask.astype(bool)

    if gt_mask.ndim == 3 and gt_mask.shape[0] > 1:
        gt_mask = np.argmax(gt_mask[1:], axis=0).astype(bool)
    else:
        gt_mask = gt_mask.astype(bool)

    try:
        if synth_mask.any() and corr_mask.any():
            pseudo_dsc = dc(corr_mask, synth_mask)
            try:
                pseudo_hd95 = hd95(corr_mask, synth_mask)
            except:
                pseudo_hd95 = 200
        else:
            pseudo_dsc = 0.0 if not (synth_mask.any() and corr_mask.any()) else np.nan
            pseudo_hd95 = 200

        if corr_mask.any() and gt_mask.any():
            real_dsc = dc(corr_mask, gt_mask)
            try:
                real_hd95 = hd95(corr_mask, gt_mask)
            except:
                real_hd95 = 200
        else:
            real_dsc = 0.0 if not (corr_mask.any() and gt_mask.any()) else np.nan
            real_hd95 = 200

        return {
            'batch_id': batch_id,
            'sample_idx': sample_idx,
            'slice': slice_idx,
            'corruption_id': corruption_id,
            'pseudo_dsc': pseudo_dsc,
            'pseudo_hd95': pseudo_hd95,
            'real_dsc': real_dsc,
            'real_hd95': real_hd95
        }
    except:
        return {
            'batch_id': batch_id,
            'sample_idx': sample_idx,
            'slice': slice_idx,
            'corruption_id': corruption_id,
            'pseudo_dsc': np.nan,
            'pseudo_hd95': 200,
            'real_dsc': np.nan,
            'real_hd95': 200
        }


def setup_model_components(args, device, loader):
    """
    Setup and load all model components needed for evaluation.
    
    Args:
        args: Configuration arguments
        device: PyTorch device
    
    Returns:
        Dictionary containing all model components
    """
    
    # Load models
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path, weights_only=True))
    
    diffusion_model = define_instance(args, "diffusion_def").to(device)
    xa = CLIPCrossAttentionGrid(output_dim=512, grid_reduction='column_softmax').to(device)
    
    trained_embed_path = os.path.join(args.model_dir, "embed_last.pt")
    #try:
    #    embed = torch.nn.Embedding(2, 512).to(device)  # Assuming slice_ratio is a single value
    #    embed.load_state_dict(torch.load(trained_embed_path, weights_only=True))
    #except:
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), 
        torch.nn.GELU(), 
        torch.nn.Linear(32, 512)
    ).to(device)
    embed.load_state_dict(torch.load(trained_embed_path, weights_only=True))
    
    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet_last.pt")
    trained_xa_path = os.path.join(args.model_dir, "xa_last.pt")
    
    diffusion_model.load_state_dict(torch.load(trained_diffusion_path, weights_only=True))
    xa.load_state_dict(torch.load(trained_xa_path, weights_only=True))
    
    logging.info(f"Loaded models from {trained_g_path}, {trained_diffusion_path}, {trained_xa_path}, {trained_embed_path}")
    
    # Setup scheduler and inferer
    scheduler = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=args.NoiseScheduler["clip_sample"],
    )
    
    check_data = first(loader)
    if args.num_classes > 1:
        check_data['label'] = ohe(check_data["label"], args.num_classes)
    z = autoencoder.encode_stage_2_inputs(check_data['label'].to(device))
    print(f"Latent feature shape {z.shape}")
    print(f"Scaling factor set to {1/torch.std(z)}\n")
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1/torch.std(z))
    
    return {
        'autoencoder': autoencoder,
        'diffusion_model': diffusion_model,
        'xa': xa,
        'embed': embed,
        'scheduler': scheduler,
        'inferer': inferer
    }


def evaluate_validation_set(args):
    """
    Main function to evaluate the entire validation set and save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    
    if args.is_msd:
        from utils import prepare_msd_dataloader
        train_l, val_loader = prepare_msd_dataloader(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            spacing=(1.0, 1.0, 1.0),
            sample_axis=args.sample_axis,
            randcrop=True,
            cache=1.0,
            download=args.download,
            size_divisible=size_divisible,
        )
    else:
        from utils import prepare_general_dataloader
        train_l, val_loader = prepare_general_dataloader(
            args,
            args.image_pattern,
            args.label_pattern,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            spacing=(1.0, 1.0, 1.0),
            sample_axis=args.sample_axis,
            randcrop=True,
            cache=1.0,
            size_divisible=size_divisible,
        )
    
    model_components = setup_model_components(args, device, train_l)
    
    # Compute metrics
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_figure_dir = os.path.join(args.output_dir, f"visualizations_{timestamp}")
    output_segmentation_dir = os.path.join(args.output_dir, f"segmentations_{timestamp}")
    os.makedirs(output_figure_dir, exist_ok=True)
    os.makedirs(output_segmentation_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, f"validation_metrics_{timestamp}.csv")
    
    df_results = compute_metrics_for_validation(
        model_components=model_components,
        val_loader=val_loader,
        args=args,
        output_figure_dir=output_figure_dir,
        output_segmentation_dir=output_segmentation_dir,
        output_csv_path=output_path
    )
    
    df_agg = df_results.groupby(["batch_id", "corruption_id"])[["pseudo_dsc", "pseudo_hd95", "real_dsc", "real_hd95"]].mean().reset_index()
    df_agg = df_agg[df_agg["real_dsc"] < 0.9] 

    print("\n=== Validation Metrics Summary ===")
    print(f"Total evaluations: {len(df_results)}")
    print(f"Unique batches: {df_results['batch_id'].nunique()}")
    print(f"Total samples: {len(df_results.groupby(['batch_id', 'sample_idx']))}")
    print("\nPseudo metrics (Generated vs Corrupted):")
    print(f"  Mean DSC: {df_agg['pseudo_dsc'].mean():.4f} ± {df_agg['pseudo_dsc'].std():.4f}")
    print(f"  Mean HD95: {df_agg['pseudo_hd95'].mean():.4f} ± {df_agg['pseudo_hd95'].std():.4f}")
    print("\nReal metrics (Corrupted vs Ground Truth):")
    print(f"  Mean DSC: {df_agg['real_dsc'].mean():.4f} ± {df_agg['real_dsc'].std():.4f}")
    print(f"  Mean HD95: {df_agg['real_hd95'].mean():.4f} ± {df_agg['real_hd95'].std():.4f}")
    if len(df_agg) > 1:
        corr_coef, p_value = pearsonr(df_agg['pseudo_dsc'], df_agg['real_dsc'])
        print(f"\nPearson correlation between pseudo DSC and real DSC: {corr_coef:.4f}. P-value: {p_value:.4f}")
        
        corr_coef_hd95, p_value = pearsonr(df_agg['pseudo_hd95'], df_agg['real_hd95'])
        print(f"Pearson correlation between pseudo HD95 and real HD95: {corr_coef_hd95:.4f}. P-value: {p_value:.4f}")
    else:
        corr_coef = np.nan
        print("Not enough data to compute Pearson correlation.")
    
    # Save summary statistics
    summary_path = os.path.join(args.output_dir, f"validation_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=== Validation Metrics Summary ===\n")
        f.write(f"Total evaluations: {len(df_results)}\n")
        f.write(f"Unique batches: {df_results['batch_id'].nunique()}\n")
        f.write(f"Total samples: {len(df_results.groupby(['batch_id', 'sample_idx']))}\n")
        f.write("\nPseudo metrics (Generated vs Corrupted):\n")
        f.write(f"  Mean DSC: {df_agg['pseudo_dsc'].mean():.4f} ± {df_agg['pseudo_dsc'].std():.4f}\n")
        f.write(f"  Mean HD95: {df_agg['pseudo_hd95'].mean():.4f} ± {df_agg['pseudo_hd95'].std():.4f}\n")
        f.write("\nReal metrics (Corrupted vs Ground Truth):\n")
        f.write(f"  Mean DSC: {df_agg['real_dsc'].mean():.4f} ± {df_agg['real_dsc'].std():.4f}\n")
        f.write(f"  Mean HD95: {df_agg['real_hd95'].mean():.4f} ± {df_agg['real_hd95'].std():.4f}\n")
        corr_coef, p_value = pearsonr(df_agg['pseudo_dsc'], df_agg['real_dsc'])
        f.write(f"\nPearson correlation between pseudo DSC and real DSC: {corr_coef:.4f}. P-value: {p_value:.4f}")
        
        corr_coef_hd95, p_value = pearsonr(df_agg['pseudo_hd95'], df_agg['real_hd95'])
        f.write(f"\nPearson correlation between pseudo HD95 and real HD95: {corr_coef_hd95:.4f}. P-value: {p_value:.4f}")
    
    return df_agg


def run_eval():
    parser = argparse.ArgumentParser(description="Medical Segmentation Validation Evaluation")
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
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="number of generated images",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    return evaluate_validation_set(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    df_agg = run_eval()