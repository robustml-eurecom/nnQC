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

import torchvision
from utils.dataset import get_transforms, get_dataloader
from models.networks import (
    ImageAutoEncoder, 
    LargeImageAutoEncoder,
    ConvAE,
    SpatialAE,
    get_device,
    MaskLoss, ImageLoss
)
from models.trainers import AETrainer, AutoencoderKLTrainer

from generative.networks.nets import PatchDiscriminator
from monai.data import DataLoader, create_test_image_3d
from monai.inferers import SliceInferer
from monai.metrics import DiceMetric
from monai.visualize import matshow3d

#monai.config.print_config()
monai.utils.set_determinism(0)     


def get_model(name, params):
    if name == "z3-img-ae":
        return ImageAutoEncoder(**params)
    elif name == "img-ae":
        return LargeImageAutoEncoder(**params)
    elif name == "mask-ae":
        return ConvAE(**params)
    elif name == "spatial-ae":
        return {
            'gen': SpatialAE(**params['generator']), 
            'dis': PatchDiscriminator(**params['discriminator'])
        }


def get_loss_fn(name, fn, opts):
    if name == "img-ae":
        return ImageLoss(fn)
    elif name == "mask-ae":
        return MaskLoss(fn, **opts['loss_args'])


def run(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    dataset_opts = config["dataset"]
    model_opts = config["ae"][args.model]
    loss_fn = config["ae"][args.model].get('loss_functions', [])
    keys = model_opts.get('keys', []) if 'keys' in model_opts else None
    
    outputs = args.model + '_outputs'
    if args.experiment_name:
        outputs = os.path.join(args.experiment_name, outputs)
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(os.path.join(outputs, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(outputs, args.ckpt_folder), exist_ok=True)
    
    device = get_device()
    model = get_model(args.model, model_opts)
    model = model.to(device) if 'spatial' not in args.model else {k: v.to(device) for k, v in model.items()}

    if 'spatial' not in args.model:
        trainer_opts = config["ae_trainer"]['standard'] 
        optimizer = torch.optim.Adam(model.parameters(), lr=model_opts['lr'])
    else:
        trainer_opts = config["ae_trainer"]['kl']
    
    if os.listdir(os.path.join(outputs, args.ckpt_folder)):
        best_ckpt = [f for f in os.listdir(os.path.join(outputs, args.ckpt_folder)) if re.search(r'best', f)][0]
        ckpt = torch.load(os.path.join(outputs, args.ckpt_folder, best_ckpt))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    os.makedirs(os.path.join(outputs, args.ckpt_folder), exist_ok=True)
    if 'spatial' in args.model:
        trainer = AutoencoderKLTrainer(
            model['gen'],
            model['dis'],
            device
        )
    else:
        trainer = AETrainer(
            model, 
            optimizer, 
            get_loss_fn(args.model, loss_fn, model_opts),
            keys,
            device,
            mode=args.model
        )
    
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
    
    trainer.train(
        train_loader=train_loader, 
        val_loader=val_loader,
        ckpt_folder=os.path.join(outputs, args.ckpt_folder),
        logs_folder=os.path.join(outputs, 'logs'),
        **trainer_opts,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--config', '-c', default='config.yml', type=str, help='Path to config file')
    parser.add_argument('--model', '-m', default='img-ae', type=str, help='Model to train')
    parser.add_argument('--data_dir', '-d', default='preprocessed', type=str, help='Path to data directory')
    parser.add_argument('--ckpt_folder', '-ckpt', default='checkpoints', type=str, help='Folder to save checkpoints')
    parser.add_argument('--experiment_name', '-e', default=None, type=str, help='Name of experiment')
    args = parser.parse_args()
    
    run(args)
    
        

    
