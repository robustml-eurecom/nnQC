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
import shutil

from torch.amp import autocast
from torchvision.models import (
    resnet18,
    resnet50
)
from utils.dataset import get_dataloader, get_transforms
from models.networks import (
    LargeImageAutoEncoder, 
    ConvAE,
    SpatialAE,
    CLIPFeatureExtractor,
    ResNetFeatureExtractor,
    get_device,
    load_checkpoint
)
from models.trainers import LDMTrainer

#from generative.networks.nets import DiffusionModelUNet
from models.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer
import warnings

warnings.filterwarnings("ignore")
monai.utils.set_determinism(0)     

def run(args):
    #open config yaml file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    outputs = 'ldm_outputs'
    if args.experiment_name:
        outputs = os.path.join(args.experiment_name, outputs)
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(os.path.join(outputs, args.log_folder), exist_ok=True)
    os.makedirs(os.path.join(outputs, args.ckpt_folder), exist_ok=True)
    
    img_model_opts = config["ae"]['img-ae']
    mask_model_opts = config["ae"]['mask-ae']
    spatial_model_opts = config["ae"]['spatial-ae']
    dataset_opts = config["dataset"]
    
    device = get_device()
    
    spatial_ae = load_checkpoint(
        torch.nn.DataParallel(SpatialAE(**spatial_model_opts['generator']), device_ids=device[1]),
        os.path.join(
            args.experiment_name,
            spatial_model_opts['ckpt_path'])
        )
    
    projector = False
    if args.train_mode == 'std' or args.train_mode == 'conv':
        feature_extractor = load_checkpoint(
            torch.nn.DataParallel(LargeImageAutoEncoder(**img_model_opts), device_ids=device[1]), 
            os.path.join(
                args.experiment_name,
                img_model_opts['ckpt_path'])
            )
        mask_ae = load_checkpoint(
            torch.nn.DataParallel(ConvAE(**mask_model_opts), device_ids=device[1]),
            os.path.join(
                args.experiment_name,
                mask_model_opts['ckpt_path'])
        )
        if args.train_mode == 'conv':
            projector = True
            
    elif args.train_mode == 'pretrained':
        feature_extractor = CLIPFeatureExtractor(None, dataset_opts['classes'])
        mask_ae = ResNetFeatureExtractor('resnet18', dataset_opts['classes'])  
    
    ldm_opts = config["ldm"]
    unet = DiffusionModelUNet(**ldm_opts['unet'])
    unet = torch.nn.DataParallel(unet, device_ids=device[1])
    scheduler = DDPMScheduler(**ldm_opts['ddpm'])     
    
    feature_extractor = feature_extractor.to(device[0])
    mask_ae = mask_ae.to(device[0])
    spatial_ae = spatial_ae.to(device[0])
    unet = unet.to(device[0])
    
    train_loader = get_dataloader(
        args.data_dir, 
        "train", 
        dataset_opts['batch_size'], 
        dataset_opts['classes'], 
        sanity_check=True
    )
    val_loader = get_dataloader(
        args.data_dir, 
        "val", 
        dataset_opts['batch_size'], 
        dataset_opts['classes'], 
        sanity_check=True
    )
    
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True):
            check_data = monai.utils.misc.first(val_loader) 
            z = spatial_ae.module.autoencoderkl.encode_stage_2_inputs(
                check_data["seg"].to(device[0])
            )
            recon, _, _ = spatial_ae(check_data["seg"].to(device[0]))
            
    plt.imshow(recon[0].argmax(0).cpu().numpy())
    plt.savefig('recon.png')
            
    print(f"Scaling factor set to {1/torch.std(z)}")
    print("Finished setup")
    print("Starting training------------------------------------------")
    
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    
    epoch = 0
    if os.listdir(os.path.join(outputs, args.ckpt_folder)):
        ckpt = torch.load(os.path.join(outputs, args.ckpt_folder, 'best.pth'))
        unet.load_state_dict(ckpt['unet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        print("Loaded checkpoint")
    
    ldm_trainer_opts = config["ldm_trainer"]
    num_steps = ldm_trainer_opts['epochs'] * sum(1 for _ in train_loader)
    ldm_trainer_opts['gamma'] = (ldm_trainer_opts['lr_end'] / ldm_trainer_opts['lr_start'])**(1/num_steps)
    trainer = LDMTrainer(
        unet=unet,
        spatial_ae=spatial_ae,
        feat_extractor=feature_extractor,
        mask_ae=mask_ae,
        inferer=inferer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device[0]
    )
            
    trainer.train(
        os.path.join(outputs, args.ckpt_folder),
        os.path.join(outputs, args.log_folder),
        projector,
        **ldm_trainer_opts
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ldm')
    parser.add_argument('--config', '-c', default='config.yml', type=str, help='Path to config file')
    parser.add_argument('--data_dir', '-d', default='preprocessed', type=str, help='Path to data directory')
    parser.add_argument('--ckpt_folder', '-ckpt', default='checkpoints', type=str, help='Folder to save checkpoints')
    parser.add_argument('--train_mode', '-t', default='std', type=str, help='Training mode')
    parser.add_argument('--experiment_name', '-e', default=None, type=str, help='Name of experiment')
    parser.add_argument('--log_folder', '-l', default='logs', type=str, help='Folder to save logs')
    args = parser.parse_args()
    
    run(args)
    
        

    
