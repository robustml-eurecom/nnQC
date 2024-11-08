import torch.nn as nn
import torch


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