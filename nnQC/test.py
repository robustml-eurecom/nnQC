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
from utils.dataset import get_dataloader, get_transforms
from utils.evaluation import ldm_testing
from models.networks import (
    LargeImageAutoEncoder, 
    ConvAE,
    SpatialAE,
    get_device
)
from models.trainers import LDMTrainer
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType
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
    test_loader = get_dataloader(
        args.data_dir, 
        "test", 
        dataset_opts['keyword'], 
        1, 
        dataset_opts['classes'], 
        get_transforms("test", dataset_opts['classes']),
        sanity_check=True
    )
    
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = monai.utils.misc.first(test_loader) 
            z = spatial_ae.autoencoderkl.encode_stage_2_inputs(
                check_data["seg"].to(device)
            )

    print(f"Scaling factor set to {1/torch.std(z)}")
    print("Finished setup")
    print("Start testing------------------------------------------")
    
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor)
    
    os.makedirs(args.results_folder, exist_ok=True)
    
    test_loaders = {}

    transform_monai = Compose([   
        Resize(spatial_size=(128, 128)),
        ScaleIntensity(),
        EnsureType()         
    ])

    transform = torchvision.transforms.Compose([
        Resizer((128,128)),
        AddPadding((256,256)),
        CenterCrop((256,256)),
        OneHot(dataset_opts['classes']),
        ToTensor()
    ])
    
    test_loaders_img = TestDataLoader(
        f"{args.data_dir}/img_testing", 
        patient_ids=range(dataset_opts['id_start']+args.start_idx, dataset_opts['id_end']+args.start_idx), 
        batch_size=2,
        transform=transform_monai,
        isimg=True
        )
    test_loaders = TestDataLoader(
        f"{args.data_dir}/testing", 
        patient_ids=range(dataset_opts['id_start']+args.start_idx, dataset_opts['id_end']+args.start_idx), 
        batch_size=2,
        transform=transform
        )
    gt_test_loaders = TestDataLoader(
        f"{args.data_dir}/testing", 
        patient_ids=range(dataset_opts['id_start']+args.start_idx, dataset_opts['id_end']+args.start_idx), 
        batch_size=2,
        transform=transform
        )
    
    #for model in sorted(test_loaders.keys()):
    #print("Processing segmentation model", model)
    
    keys = [f'Class {i}' for i in range(dataset_opts['classes'])]
    results = ldm_testing(
        unet, 
        spatial_ae,
        mask_ae,
        feature_extractor,
        inferer,
        scheduler,
        [test_loaders_img, test_loaders, gt_test_loaders],
        keys,
        args.results_folder
    )
    
    results_output = os.path.join(args.results_folder, "measures")
    os.makedirs(results_output, exist_ok=True)
    np.save("{}/ldm_scores.npy".format(results_output), results['ldm'])
    np.save("{}/real_scores.npy".format(results_output), results['gt'])
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
    
        

    
