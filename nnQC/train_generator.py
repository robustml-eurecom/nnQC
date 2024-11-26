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

from utils.dataset import get_dataloader, get_transforms
from nnQC.models.networks import (
    ImageAutoEncoder, 
    LargeImageAutoEncoder,
    MaskAutoEncoder, 
    ConvAE,
    get_device
)
from models.gans import (
    Generator,
    Discriminator,
    Generator_v1
)
from models.trainers import GANTrainer

#monai.config.print_config()
monai.utils.set_determinism(0)     

def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(os.path.join(checkpoint_path, 'best_model.pth'))
    model.load_state_dict(ckpt)
    return model

def run(args):
    #open config yaml file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    img_model_opts = config["ae"]['img-ae']
    mask_model_opts = config["ae"]['mask-ae']
    #feature_extractor = load_checkpoint(ImageAutoEncoder(**img_model_opts), config["ae"]['img-ae']['ckpt_path'])
    feature_extractor = load_checkpoint(LargeImageAutoEncoder(**img_model_opts), config["ae"]['img-ae']['ckpt_path'])
    #mask_ae = load_checkpoint(MaskAutoEncoder(**mask_model_opts), config["ae"]['mask-ae']['ckpt_path'])
    mask_ae = load_checkpoint(ConvAE(**mask_model_opts), config["ae"]['mask-ae']['ckpt_path'])
    
    outputs = 'gan_outputs'
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(os.path.join(outputs, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(outputs, args.ckpt_folder), exist_ok=True)
    
    device = get_device()
    #generator = Generator(3*2, 32, 3)
    generator = Generator_v1(3)
    discriminator = Discriminator(3)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    feature_extractor = feature_extractor.to(device)
    mask_ae = mask_ae.to(device)
    
    trainer_opts = config["gan_trainer"]
    loss_fn = trainer_opts["loss_args"]
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=trainer_opts['lr'], betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=trainer_opts['lr'], betas=(0.5, 0.999))   
    
    if os.listdir(os.path.join(outputs, args.ckpt_folder)):
        ckpt = torch.load(os.path.join(outputs, args.ckpt_folder, 'best_model.pth'))
        generator.load_state_dict(ckpt['G'])
        discriminator.load_state_dict(ckpt['D'])
        g_optimizer.load_state_dict(ckpt['G_optim'])
        d_optimizer.load_state_dict(ckpt['D_optim'])
    
    trainer = GANTrainer(
        generator, 
        discriminator, 
        g_optimizer, 
        d_optimizer,
        device,
        feature_extractor,
        mask_ae,
        loss_fn,
        args.lambda_gan
    )
    
    dataset_opts = config["dataset"]
    train_loader = get_dataloader(
        args.data_dir, 
        "train", 
        dataset_opts['keyword'], 
        dataset_opts['batch_size'], 
        dataset_opts['classes'], 
        get_transforms("train", dataset_opts['classes']),
        sanity_check=True
    )
    
    val_loader = get_dataloader(
        args.data_dir, 
        "val", 
        dataset_opts['keyword'], 
        dataset_opts['batch_size'], 
        dataset_opts['classes'], 
        get_transforms("val", dataset_opts['classes']),
        sanity_check=True
    )
        
    trainer.train(
        train_loader, 
        val_loader,
        trainer_opts['epochs'],
        trainer_opts['val_interval'],
        os.path.join(outputs, args.ckpt_folder),
        os.path.join(outputs, 'logs')
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--config', '-c', default='config.yml', type=str, help='Path to config file')
    parser.add_argument('--data_dir', '-d', default='preprocessed', type=str, help='Path to data directory')
    parser.add_argument('--ckpt_folder', '-ckpt', default='checkpoints', type=str, help='Folder to save checkpoints')
    parser.add_argument('--lambda_gan', '-l', default=0.1, type=float, help='GAN loss weight')
    
    args = parser.parse_args()
    
    run(args)
    
        

    
