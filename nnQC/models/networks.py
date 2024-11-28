import numpy as np
import os
import re
import torch
import torch.optim as optim
import torch.nn as nn

from monai.networks.nets import AutoEncoder
from monai.networks.layers.factories import Act
from generative.networks.nets import AutoencoderKL

import timm

from transformers import CLIPProcessor, CLIPModel

import sys;sys.path.append("/data/marciano/experiments/pull-QC/nnQC/nnQC/models/pytorch-ssim")
import pytorch_ssim


def get_device():
    #return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #return usage of CUDA_AVAILABLE_DEVICES=0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_checkpoint(model, checkpoint_path):
    ckpt = [f for f in os.listdir(checkpoint_path) if re.search(r'best', f)]
    if isinstance(model, SpatialAE):
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, ckpt[0]))['autoencoderkl_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, ckpt[0]))['model'])
    print(f'Loaded checkpoint from {os.path.join(checkpoint_path, ckpt[0])}')
    return model


class LargeImageAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(LargeImageAutoEncoder, self).__init__()
        self.feature_extractor = AutoEncoder(
            spatial_dims=2,
            in_channels=kwargs['n_channels'],
            out_channels=kwargs['n_channels'],
            channels=(32, 32, 64, 64, 128, 128, 256, 512, kwargs['z_dim']),
            strides=(2, 1, 2, 1, 2, 1, 2, 2, 2),
            num_res_units=4
        )
        self.sigmoid = nn.Sigmoid()

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
    
    def forward(self, x):
        output = self.feature_extractor(x)
        return self.sigmoid(output)
    
    def encode(self, x):
        return self.feature_extractor.encode(x)
    

class ImageAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(ImageAutoEncoder, self).__init__()
        self.feature_extractor = AutoEncoder(
            spatial_dims=2,
            in_channels=kwargs['n_channels'],
            out_channels=kwargs['n_channels'],
            channels=(64, 128, 256, kwargs['z_dim']),
            kernel_size=4,
            strides=[2,2,2,2],
            num_inter_units=0,
            act=Act.LEAKYRELU,
            padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
    
    def forward(self, x):
        output = self.feature_extractor(x)
        return self.sigmoid(output)
    
    def encode(self, x):
        return self.feature_extractor.encode(x)
    
    
class MaskAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(MaskAutoEncoder, self).__init__()
        self.feature_extractor = AutoEncoder(
            spatial_dims=2,
            in_channels=kwargs['n_channels'],
            out_channels=kwargs['n_channels'],
            channels=(64, 128, 256, kwargs['z_dim']),
            kernel_size=4,
            strides=[2,2,2,2],
            num_inter_units=0,
            act=Act.LEAKYRELU,
            padding=1
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        output = self.feature_extractor(x)
        return self.softmax(output)
    
    def encode(self, x):
        return self.feature_extractor.encode(x)
    

class ConvAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_layers(kwargs["z_dim"], kwargs['n_channels'])
        self.apply(self.weight_init)
        self.n_classes = kwargs['n_channels']

    def init_layers(self, latent_size, n_classes):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=n_classes, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=latent_size, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=n_classes, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decoder(latent)
        latent = latent.view(latent.size(0), -1)
        return reconstruction


class SpatialAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels):
        super().__init__()
        self.autoencoderkl = AutoencoderKL(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=(64, 128, 256),
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
        self.softmax = nn.Softmax(dim=1)    
        
    def forward(self, x):
        x, z_mu, z_sigma = self.autoencoderkl(x)
        x = self.softmax(x)
        return x, z_mu, z_sigma
    
    def encode_stage_2_inputs(self, x):
        z,_ = self.autoencoderkl.encode(x)
        return z
    
    def decode_stage_2_outputs(self, z):
        x = self.autoencoderkl.decode(z)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leakyrelu'):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.upconv_expanding = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(True)
        
        upconv = [self.upconv, self.upconv_expanding, self.norm, self.activation]
        self.upconv_block = nn.Sequential(*upconv)
        
    def forward(self, x):
        return self.upconv_block(x)    
        
    
class Generator_v1(nn.Module):
    def __init__(self, n_channels):
        super(Generator_v1, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 200,  512),
            nn.ReLU(True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256 * 16 * 16),
            nn.ReLU(True)
        )
                
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2, True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2, True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2, True),
            nn.ConvTranspose2d(32, n_channels, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1) 
        )

    def forward(self, features, embeddings):
        # Flatten the input tensors
        features = features.view(features.size(0), -1)
        embeddings = embeddings.view(embeddings.size(0), -1)

        # Concatenate features and embeddings
        x = torch.cat((features, embeddings), dim=1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 256, 16, 16)

        # Pass through deconvolution layers
        generated_mask = self.deconv_layers(x)

        return generated_mask
    

class Generator(nn.Module):
    def __init__(self, z_dim, ngf, n_channels):
        super(Generator, self).__init__()

        layers = []
        in_channels = ngf * 8
        out_channels = in_channels // 2

        # First upconv layer
        layers.append(nn.ConvTranspose2d(z_dim, in_channels, kernel_size=3, stride=1, padding=1))

        # Iteratively create deconvolutional layers
        while out_channels >= 32:
            layers.append(UpConvBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels //= 2

        # Last deconv layer (not an upconv)
        layers.append(nn.ConvTranspose2d(in_channels, n_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Softmax(dim=1))

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, features, embeddings):
        # Concatenate features and embeddings
        x = torch.cat((features, embeddings), dim=1)
        generated_mask = self.deconv_layers(x)
        return generated_mask   


#AE Generator version
class AEGenerator(nn.Module):
    def __init__(self, n_channels):
        super(AEGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels+1, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 100, kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, n_channels, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1)
        )
        
    def forward(self, mask):
        x = self.conv_layers(mask)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LatentPerceptualLoss(nn.Module):
    def __init__(self):
        super(LatentPerceptualLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, original_latent, reconstructed_latent):
        return self.mse_loss(reconstructed_latent, original_latent)

class CycleConsistencyLoss(nn.Module):
    def __init__(self, ae):
        super(CycleConsistencyLoss, self).__init__()
        self.ae = ae
        self.mse_loss = nn.MSELoss()

    def forward(self, original_latent, generated_mask):
        # Encode the generated mask back into the latent space
        reconstructed_latent = self.ae.encode(generated_mask)
        # Compute the MSE loss between the original and reconstructed latent codes
        return self.mse_loss(reconstructed_latent, original_latent)


class BKGDLoss:
    def __call__(self, prediction, target):
        intersection = torch.sum(prediction * target, dim=(1,2,3))
        cardinality = torch.sum(prediction + target, dim=(1,2,3))
        dice_score = 2. * intersection / (cardinality + 1e-6)
        return torch.mean(1 - dice_score)


class GDLoss:
    def __call__(self, x, y):
        tp = torch.sum(x * y, dim=(0,2,3))
        fp = torch.sum(x * (1-y), dim=(0,2,3))
        fn = torch.sum((1-x) * y, dim=(0,2,3))
        nominator = 2*tp + 1e-06
        denominator = 2*tp + fp + fn + 1e-06
        dice_score =- (nominator / (denominator+1e-6))[1:].mean()
        return dice_score


class TI_Loss(torch.nn.Module):
    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2 : 
            self.sum_dim_list = [1,2,3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3 :
            self.sum_dim_list = [1,2,3,4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True) # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False) # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)


    def set_kernel(self):
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                                        [[0,0,0],[0,1,0],[0,0,0]],
                                        [[0,1,0],[1,1,1],[0,1,0]],
                                        [[0,0,0],[0,1,0],[0,0,0]]
                                    ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))
        
        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel,axis=0), axis=0))


    def topological_interaction_module(self, P):
        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
            
            # Get Neighbourhood Information
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding='same')
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding='same')
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map


    def forward(self, x, y):
        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(),dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(),y[:,0].long()),dim=1)
        ce_tensor[:,0] = ce_tensor[:,0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value

    
class MaskLoss:
    def __init__(self, functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss):
        self.MSELoss = nn.MSELoss()
        self.BKMSELoss = nn.MSELoss()
        self.BKGDLoss = BKGDLoss()
        self.GDLoss = GDLoss()
        self.TI_Loss = TI_Loss(dim=2, connectivity=4, inclusion=[], exclusion=[[1,2]])
        self.functions = functions
        self.settling_epochs_BKGDLoss = settling_epochs_BKGDLoss
        self.settling_epochs_BKMSELoss = settling_epochs_BKMSELoss


    def __call__(self, prediction, target, epoch, validation=False):
        contributes = {}
        for f in self.functions:
            if f == 'MSELoss':
                contributes[f] = self.MSELoss(prediction[:,1:], target[:,1:])
            elif f == 'BKMSELoss':
                contributes[f] = self.BKMSELoss(prediction, target)
            elif f == 'BKGDLoss':
                contributes[f] = self.BKGDLoss(prediction, target)
            elif f == 'GDLoss':
                contributes[f] = self.GDLoss(prediction, target)
            elif f == 'TI_Loss':
                target_argmax = torch.argmax(target, dim=1, keepdim=True)
                contributes[f] = 1e-4 * self.TI_Loss(prediction, target_argmax)

        if "BKGDLoss" in contributes and epoch < self.settling_epochs_BKGDLoss:
            contributes["BKGDLoss"] += self.BKGDLoss(prediction[:,1:], target[:,1:])
        if "BKMSELoss" in contributes and epoch < self.settling_epochs_BKMSELoss:
            contributes["BKMSELoss"] += self.BKMSELoss(prediction[:,1:], target[:,1:])
        
        contributes["Total"] = sum(contributes.values())
        
        if validation:
            return {k: v.item() for k,v in contributes.items()}
        else:
            return contributes["Total"]


class ImageLoss:
    def __init__(self, functions):
        self.MSELoss = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.functions = functions

    def __call__(self, prediction, target, epoch):
        contributes = {}
        for f in self.functions:
            if f == 'MSELoss':
                contributes[f] = self.MSELoss(prediction, target)
            elif f == 'SSIM':
                contributes[f] = 1 - self.SSIM(prediction, target)
        contributes["Total"] = sum(contributes.values())
        return contributes["Total"]

def ae_full_loss(recon, gt):
    recon_image = recon[:, 0, :, :].unsqueeze(1)
    recon_mask = recon[:, 1:, :, :]
    gt_image = gt[:, 0, :, :].unsqueeze(1)
    gt_mask = gt[:, 1:, :, :]
    img_recon_loss = pytorch_ssim.SSIM()(recon_image, gt_image)
    mask_recon_loss = nn.CrossEntropyLoss()(recon_mask, gt_mask) + nn.MSELoss()(recon_mask, gt_mask)
    return img_recon_loss + mask_recon_loss


class InfererWrapper(nn.Module):
    def __init__(self, inferer):
        super(InfererWrapper, self).__init__()
        self.inferer = inferer

    def sample(self, x, **kwargs):
        return self.inferer.sample(x, **kwargs)
    
    def forward(self, x, **kwargs):
        return self.inferer.sample(x, **kwargs)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, text, normalize=True):
        super(CLIPFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text = text
        self.normalize = normalize
        
    def encode(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
            
        inputs = self.processor(
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        feat = self.model.get_image_features(**inputs.to("cuda"))
        return torch.nn.functional.normalize(feat, p=2, dim=1)
    
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone, n_classes, normalize=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=1)
        self.n_classes = n_classes
        self.normalize = normalize  
        self.model = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def encode(self, image):
        if image.shape[1] > 1:
            image = torch.argmax(image, dim=1).unsqueeze(1)
            # scale by 255
            image = image/(self.n_classes-1) * 255
            # scale to 0-1  
            image = image/255
            
        return torch.nn.functional.normalize(self.model(image), p=2, dim=1) if self.normalize else self.model(image)

        