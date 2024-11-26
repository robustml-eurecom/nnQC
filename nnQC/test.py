import os
import tempfile
from glob import glob
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
import yaml
import argparse
import re

from torch.cuda.amp import autocast
from utils.dataset import get_test_dataloader, get_transforms
from utils.evaluation import ldm_testing
from models.networks import (
    LargeImageAutoEncoder, 
    ConvAE,
    SpatialAE,
    InfererWrapper,
    get_device
)
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType, SpatialPad, AsDiscrete, EnsureChannelFirst
from utils.dataset import TestDataLoader, AddPadding, CenterCrop, OneHot, ToTensor, Resizer
import random
import torchvision
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer

monai.utils.set_determinism(0)     


def load_checkpoint(model, checkpoint_path):
    ckpt = [f for f in os.listdir(checkpoint_path) if re.search(r'best', f)]
    if isinstance(model, SpatialAE):
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, ckpt[0]))['autoencoderkl_state_dict'])
    elif isinstance(model, DiffusionModelUNet):
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, ckpt[0]))['unet_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, ckpt[0]))['model'])
    print(f'Loaded checkpoint from {os.path.join(checkpoint_path, ckpt[0])}')
    return model

def run(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    inputs = 'ldm_outputs'
    inputs = os.path.join(args.experiment_name, inputs)
    ldm_ckpts = os.path.join(inputs, args.ckpt_folder)

    img_model_opts = config["ae"]['img-ae']
    spatial_model_opts = config["ae"]['spatial-ae']
    mask_model_opts = config["ae"]['mask-ae']
    
    feature_extractor = load_checkpoint(
        LargeImageAutoEncoder(**img_model_opts), 
        os.path.join(
            args.experiment_name,
            img_model_opts['ckpt_path'])
        )
    spatial_ae = load_checkpoint(
        SpatialAE(**spatial_model_opts['generator']),
        os.path.join(
            args.experiment_name,
            spatial_model_opts['ckpt_path'])
        )
    unet = load_checkpoint(
        DiffusionModelUNet(**config["ldm"]['unet']),
        ldm_ckpts
    )
    
    device = get_device()
    
    if args.dual:
        mask_ae = load_checkpoint(
            ConvAE(**mask_model_opts),
            os.path.join(
                args.experiment_name,
                mask_model_opts['ckpt_path'])
            )
        mask_ae = mask_ae.to(device)
    else:
        mask_ae = None
    
    feature_extractor = feature_extractor.to(device)
    spatial_ae = spatial_ae.to(device)
    unet = unet.to(device)
   
    ldm_opts = config["ldm"]
    scheduler = DDPMScheduler(**ldm_opts['ddpm'])
    
    dataset_opts = config["dataset"]
    sample = nib.load(glob(os.path.join(args.data_dir, "testing", "**", "*.nii.gz"))[5])
    x, y, z = nib.aff2axcodes(sample.affine)
    axcodes = x + y + z
    spacing = sample.header.get_zooms()
    
    print("Orientation: ", axcodes)
    print("Spacing: ", spacing)
    
    fingerprints = {    
        "axcodes": axcodes,
        "spacing": spacing,
        "shape": sample.shape
    }
    
    print()
    print("Processing test data from index {} to {}".format(dataset_opts['id_start'], dataset_opts['id_end']))
    test_loader = get_test_dataloader(
        args.data_dir,
        [dataset_opts['id_start'], dataset_opts['id_end']],
        dataset_opts['keyword'],
        dataset_opts['classes'], 
        fingerprints,
        True
    )
    
    gt_loader = test_loader
    
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = monai.utils.misc.first(test_loader) 
            z = spatial_ae.autoencoderkl.encode_stage_2_inputs(
                check_data["seg"].to(device)
            )

    print(f"Scaling factor set to {1/torch.std(z)}")
    print("Finished setup")
    print()
    print("Start testing")
    print()
    
    scale_factor = 1 / torch.std(z)
    ldm_inferer = LatentDiffusionInferer(scheduler, scale_factor)
    inferer = InfererWrapper(ldm_inferer)
    
    os.makedirs(args.results_folder, exist_ok=True)
      
    keys = [f'Class {i}' for i in range(dataset_opts['classes'])]
    results = ldm_testing(
        unet, 
        spatial_ae,
        mask_ae, 
        feature_extractor, 
        inferer, 
        scheduler, 
        [gt_loader, test_loader],
        dataset_opts['id_start'],
        dataset_opts['id_end'],
        keys,
        args.results_folder
    )
    
    results_output = os.path.join(args.results_folder, "measures")
    os.makedirs(results_output, exist_ok=True)
    np.save("{}/ldm_scores.npy".format(results_output), results['ldm'])
    np.save("{}/real_scores.npy".format(results_output), results['gt'])
    np.save("{}/baseine.npy".format(results_output), results['baseline'])
    print("-----------------------------------")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ldm')
    parser.add_argument('--config', '-c', default='config.yml', type=str, help='Path to config file')
    parser.add_argument('--data_dir', '-d', default='preprocessed', type=str, help='Path to data directory')
    parser.add_argument('--start_idx', '-s', default=0, type=int, help='Start index for patient ids')
    parser.add_argument('--ckpt_folder', '-ckpt', default='checkpoints', type=str, help='Folder to save checkpoints')
    parser.add_argument('--dual', action='store_true', help='Use Dual CrossAttn mechanism')
    parser.add_argument('--experiment_name', '-e', default=None, type=str, help='Name of experiment')
    parser.add_argument('--results_folder', '-r', default='results', type=str, help='Folder to save results')
    
    args = parser.parse_args()
    
    run(args)
    
        

    
