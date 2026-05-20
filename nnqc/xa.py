import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor, AutoProcessor
from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
from nnqc.visualize import visualize_2d_image
import os
import cv2
import numpy as np
#import clip

class CrossAttentionGrid(nn.Module):
    """
    Creates a grid-like cross-attention between two encoders, similar to the diagram
    """
    
    def __init__(self, 
                 feature_dim_i, 
                 feature_dim_m, 
                 output_dim=512,
                 grid_reduction='column_softmax'):
        """
        Args:
            feature_dim_i: Feature dimension of first encoder (I)
            feature_dim_m: Feature dimension of second encoder (T/M)
            output_dim: Final output dimension
            grid_reduction: 'column_softmax', 'row_softmax', 'global_softmax', 'sigmoid_mean'
        """
        super(CrossAttentionGrid, self).__init__()
        
        self.grid_reduction = grid_reduction
        
        # Ensure same feature dimensions for dot product
        if feature_dim_i != feature_dim_m:
            self.proj_i = nn.Linear(feature_dim_i, feature_dim_m)
            self.use_projection = True
        else:
            self.use_projection = False
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim_m, feature_dim_m * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim_m * 2, output_dim)
        )
        
    def forward(self, I_features, T_features):
        """
        Args:
            I_features: [B, N, feature_dim_i] - Image encoder features
            T_features: [B, M, feature_dim_m] - Text/Second encoder features
        
        Returns:
            output: [B, output_dim]
            attention_grid: [B, N, M] - The attention matrix for visualization
        """
        B, N, _ = I_features.shape
        B, M, _ = T_features.shape
        
        if self.use_projection:
            I_proj = self.proj_i(I_features)  # [B, N, feature_dim_m]
        else:
            I_proj = I_features
            
        # Create the attention grid: I_i * T_j for all i,j
        # This is equivalent to I @ T^T
        attention_grid = torch.bmm(I_proj, T_features.transpose(1, 2))  # [B, N, M]
        
        if self.grid_reduction == 'column_softmax':
            # Apply softmax column-wise (across N dimension)
            attention_weights = F.softmax(attention_grid, dim=1)  # [B, N, M]
            attended_features = torch.bmm(attention_weights.transpose(1, 2), I_proj)  # [B, M, feature_dim_m]
            output_features = torch.mean(attended_features, dim=1)  # [B, feature_dim_m]
            
        elif self.grid_reduction == 'row_softmax':
            # Apply softmax row-wise (across M dimension)
            attention_weights = F.softmax(attention_grid, dim=2)  # [B, N, M]
            attended_features = torch.bmm(attention_weights, T_features)  # [B, N, feature_dim_m]
            output_features = torch.mean(attended_features, dim=1)  # [B, feature_dim_m]
            
        elif self.grid_reduction == 'global_softmax':
            # Apply softmax globally
            flat_attention = attention_grid.view(B, -1)  # [B, N*M]
            global_weights = F.softmax(flat_attention, dim=1).view(B, N, M)  # [B, N, M]
            
            # Combine features using global attention
            I_pooled = torch.sum(global_weights.sum(dim=2, keepdim=True) * I_proj, dim=1)  # [B, feature_dim_m]
            T_pooled = torch.sum(global_weights.sum(dim=1, keepdim=True).transpose(1, 2) * T_features, dim=1)  # [B, feature_dim_m]
            output_features = (I_pooled + T_pooled) / 2
            
        elif self.grid_reduction == 'sigmoid_mean':
            # Apply sigmoid and take mean
            sigmoid_grid = torch.sigmoid(attention_grid)  # [B, N, M]
            # Average across both dimensions, then use as weighting
            mean_attention = torch.mean(sigmoid_grid, dim=[1, 2], keepdim=True)  # [B, 1, 1]
            
            # Combine features
            I_pooled = torch.mean(I_proj, dim=1)  # [B, feature_dim_m]
            T_pooled = torch.mean(T_features, dim=1)  # [B, feature_dim_m]
            output_features = mean_attention.squeeze() * (I_pooled + T_pooled) / 2
            
        else:
            raise ValueError(f"Unknown grid reduction method: {self.grid_reduction}")
        
        output = self.output_proj(output_features)
        
        return output, attention_grid

class CLIPCrossAttentionGrid(nn.Module):
    """
    CLIP-based cross-attention grid that can work with:
    1. Image + Mask (both encoded with CLIP vision encoder when text is None)
    2. Image + Text (when text is provided, mask is ignored)
    """
    
    def __init__(self, 
                 clip_model_name='ViT-B-16-quickgelu',
                 clip_model_path='trained_weights/unimed_clip_vit_b16.pt',
                 text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                 output_dim=512,
                 grid_reduction='column_softmax'):
        """
        Args:
            clip_model_name: CLIP model name/path
            output_dim: Final output dimension
            grid_reduction: Attention reduction method
            use_huggingface: If True, use HuggingFace transformers, else use OpenAI CLIP
        """
        super(CLIPCrossAttentionGrid, self).__init__()
        
        self.grid_reduction = grid_reduction
        
        mean, std = get_mean_std()
        device='cuda'
    
        #self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        #self.processor = AutoProcessor.from_pretrained(clip_model_name)        
        self.unimedclip , _, self.processor = create_model_and_transforms(
            clip_model_name,
            clip_model_path,
            precision='amp',
            device=device,
            force_quick_gelu=True,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=text_encoder_name,)
        
        #self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = HFTokenizer(
            text_encoder_name,
            context_length=256,
            **{},)
            #logits = (model.logit_scale.exp() * image_features @ text_features.t()).detach().softmax(dim=-1)
        
        self.vision_feat_dim = self.text_feat_dim = 512     # 512 for base
        
        # Cross-attention between image and mask/text features
        self.cross_attention = CrossAttentionGrid(
            feature_dim_i=self.vision_feat_dim,  # Image features
            feature_dim_m=self.text_feat_dim,  # Mask features (same as image) or text features
            output_dim=output_dim,
            grid_reduction=grid_reduction
        )
        
        # Project text features to vision feature dimension if needed
        if self.text_feat_dim != self.vision_feat_dim:
            self.text_projection = nn.Linear(self.text_feat_dim, self.vision_feat_dim)
        else:
            self.text_projection = nn.Identity()
    
    def encode_image_or_mask(self, input_tensor, is_mask=False):
        """
        Extract features using CLIP vision encoder for both images and masks
        
        Args:
            input_tensor: [B, C, H, W] - RGB image tensor or mask tensor
            is_mask: Boolean indicating if this is a mask (for preprocessing)
        
        Returns:
            patch_features: [B, N, vision_feat_dim] - Patch features
            pooled_features: [B, vision_feat_dim] - Global features
        """
        # Preprocess mask if needed
        if is_mask:
            input_tensor = self.preprocess_mask_for_clip(input_tensor)
        
        # HuggingFace CLIP
        input_tensor = input_tensor.cuda()
        input_tensor = [cv2.normalize(i[0].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX) for i in input_tensor] 
        input_tensor = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in input_tensor])
        input_tensor = np.array([cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR) for img in input_tensor])  # Resize to 224x224
        input_tensor = torch.from_numpy(input_tensor).permute(0, 3, 1, 2).float().cuda()
        
        # Process inputs
        patch_features = self.unimedclip.encode_image(input_tensor)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        patch_features = patch_features.unsqueeze(1)
            
        return patch_features, None
    
    def preprocess_mask_for_clip(self, mask):
        """
        Convert mask to 3-channel format suitable for CLIP vision encoder
        
        Args:
            mask: [B, H, W] or [B, 1, H, W] or [B, C, H, W] - Mask tensor
        
        Returns:
            processed_mask: [B, 3, H, W] - 3-channel mask for CLIP
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # Handle different mask types
        if mask.shape[1] == 1:
            # Single channel mask - convert to 3 channels
            mask_rgb = np.array([visualize_2d_image(m[0]).transpose([2, 1, 0]) for m in mask])  # Convert to RGB
            mask_rgb = torch.from_numpy(mask_rgb).float().cuda()  # Convert to tensor
                
        elif mask.shape[1] == 3:
            # Already 3 channels
            mask_rgb = mask.float()
            
        else:
            # Multi-channel mask (e.g., one-hot): convert to RGB
            # Take argmax to get class indices
            class_mask = torch.argmax(mask, dim=1, keepdim=True).float()
            num_classes = mask.shape[1]
            
            # Normalize and create RGB
            mask_rgb = class_mask / (num_classes - 1) 
            
        # Ensure values are in [0, 1] range
        #mask_rgb = torch.clamp(mask_rgb, 0.0, 1.0)
        return mask_rgb
    
    
    def encode_text(self, text):
        """
        Extract text features using CLIP text encoder
        
        Args:
            text: List of strings or string
        
        Returns:
            text_features: [B, seq_len, vision_feat_dim] - Text token features
            pooled_features: [B, vision_feat_dim] - Global text features
        """
        if isinstance(text, str):
            text = [text]
        
        text = [self.tokenizer(cls_text).to(next(self.unimedclip.parameters()).device, non_blocking=True) for cls_text in text]
        text = torch.cat(text, dim=0)  # Concatenate all text inputs
        text_features = self.unimedclip.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(1).cuda()  # Add sequence dimension
        
        # Project text features to vision feature dimension
        token_features = self.text_projection(text_features)
        #pooled_features = self.text_projection(pooled_features)
        
        return token_features, None
    
    def forward(self, image, mask=None, text=None, ext_features=None):
        """
        Forward pass
        
        Args:
            image: [B, C, H, W] - Input image
            mask: [B, H, W] or [B, 1, H, W] or [B, C, H, W] - Input mask (used if text is None)
            text: List of strings or None - Input text (if provided, mask is ignored)
        
        Returns:
            output: [B, output_dim] - Final embedding
            attention_grid: [B, N, M] - Attention matrix for visualization
            features: Dict with intermediate features
        """
        # Extract image features using CLIP vision encoder
        image_features, _ = self.encode_image_or_mask(image, is_mask=False)
        
        if text is not None:
            # Use text encoder
            text_features, _ = self.encode_text(text)
            secondary_features = text_features
        elif mask is not None:
            # Use CLIP vision encoder for mask
            if mask is None:
                raise ValueError("Either text or mask must be provided")
            mask_features, _ = self.encode_image_or_mask(mask, is_mask=True)
            secondary_features = mask_features
        elif ext_features is not None:
            # Use external features directly
            if ext_features.dim() == 2:
                ext_features = ext_features.unsqueeze(1)
            secondary_features = ext_features
        else:
            raise ValueError("Either text, mask, or external features must be provided")
        
        # Apply cross-attention
        output, attention_grid = self.cross_attention(image_features.cuda(), secondary_features.cuda())
        
        return output, attention_grid, {
            'image_features': image_features,
            'secondary_features': secondary_features,
        }


# Usage example
if __name__ == "__main__":
    # Create model
    model = CLIPCrossAttentionGrid(
        output_dim=512,
        grid_reduction='column_softmax',
    )
    model = model.cuda()
    print(model)
    # Example data
    batch_size = 4
    image = torch.randn(batch_size, 1, 240, 240)  # RGB images
    image = image.clamp(0, 1)  # Ensure values are in [0, 1]
    
    # Multi-class mask (ACDC-style: 0=background, 1=RV, 2=myocardium, 3=LV)
    #mask = torch.randint(0, 4, (batch_size, 224, 224))
    
    # Binary mask alternative
    mask = torch.randint(0, 1, (batch_size, 1, 240, 240))
    
    # One-hot mask alternative  
    # mask = torch.zeros(batch_size, 4, 224, 224)
    # mask[:, 0] = 1  # All background for simplicity
    
    text = ["heart segmentation", "cardiac MRI", "medical image", "anatomical structure"]
    
    # Test with mask (both image and mask go through CLIP vision encoder)
    print("=== Testing with Image + Mask (both via CLIP Vision) ===")
    output, attention_mask, features_mask = model(image, mask=mask, text=None)
    print(f"Output shape: {output.shape}")
    print(f"Attention grid shape: {attention_mask.shape}")
    print(f"Image features shape: {features_mask['image_features'].shape}")
    print(f"Mask features shape: {features_mask['secondary_features'].shape}")
    
    # Test with text
    print("\n=== Testing with Image + Text ===")
    output, attention_text, features_text = model(image, mask=None, text=text)
    print(f"Output shape: {output.shape}")
    print(f"Attention grid shape: {attention_text.shape}")
    print(f"Image features shape: {features_text['image_features'].shape}")
    print(f"Text features shape: {features_text['secondary_features'].shape}")
    
    # Visualize attention
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mask attention (Image patches vs Mask patches)
    im1 = ax1.imshow(attention_mask[0].detach().cpu().numpy(), cmap='viridis')
    ax1.set_title('Image-Mask Cross-Attention\n(Both via CLIP Vision)')
    ax1.set_xlabel('Mask Patches')
    ax1.set_ylabel('Image Patches')
    plt.colorbar(im1, ax=ax1)
    
    # Text attention (Image patches vs Text tokens)
    im2 = ax2.imshow(attention_text[0].detach().cpu().numpy(), cmap='viridis')
    ax2.set_title('Image-Text Cross-Attention')
    ax2.set_xlabel('Text Tokens')
    ax2.set_ylabel('Image Patches')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('dummy_cross_attention_grid.png', dpi=300)