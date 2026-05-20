# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import fnmatch  
import glob
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, generate_binary_structure, label, distance_transform_edt, map_coordinates
import random
from monai.apps import DecathlonDataset
from monai.bundle import ConfigParser
from monai.data import DataLoader, CacheDataset, list_data_collate
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    SplitDimd,
    SqueezeDimd,
    AsDiscreted,
    CropForegroundd,
    Resized,
    CopyItemsd,
    RandSpatialCropd,
    Spacingd,
    RandCoarseDropoutd,
)
from monai import transforms
from monai.transforms import MapTransform, Transform
from monai.config import KeysCollection
from monai.utils import progress_bar
from typing import Dict, Hashable, Mapping, Union, Tuple, Optional, List
import numpy as np


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=36000),
        rank=rank,
        world_size=world_size,
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def radial_pinch(mask, pinch_strength=0.5, center=None):
    """
    Apply radial pinch effect that squeezes the mask toward its center.
    """
    if not mask.any():
        return mask
    
    h, w = mask.shape
    if center is None:
        center = (h // 2, w // 2)
    
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    
    if max_r > 0:
        r_normalized = r / max_r
        pinch_factor = 1 - pinch_strength * r_normalized
        pinch_factor = np.maximum(pinch_factor, 0.1)
        
        new_x = cx + (x - cx) * pinch_factor
        new_y = cy + (y - cy) * pinch_factor
        
        coords = np.array([new_y.ravel(), new_x.ravel()])
        pinched = map_coordinates(mask.astype(float), coords, order=1, mode='constant', cval=0)
        return (pinched.reshape(mask.shape) > 0.5).astype(mask.dtype)
    
    return mask

def center_of_mass_pinch(mask, pinch_strength=0.3):
    """
    Pinch toward center of mass of the segmentation.
    """
    if not mask.any():
        return mask
    
    binary_mask = mask > 0
    from scipy.ndimage import center_of_mass
    com = center_of_mass(binary_mask)
    
    if np.isnan(com).any():
        return mask
    
    return radial_pinch(mask, pinch_strength, center=(int(com[0]), int(com[1])))

def distance_based_shrink(mask, shrink_factor=0.8):
    """
    Shrink based on distance transform - removes pixels at the boundary.
    """
    if not mask.any():
        return mask
    
    binary_mask = mask > 0
    dist = distance_transform_edt(binary_mask)
    
    threshold = np.percentile(dist[dist > 0], (1 - shrink_factor) * 100)
    shrunk_binary = dist >= threshold
    
    result = np.zeros_like(mask)
    result[shrunk_binary] = mask[shrunk_binary]
    return result

def apply_progressive_pinch(mask, slice_idx, total_slices, 
                           pinch_type='radial',
                           max_pinch_strength=0.7,
                           start_fraction=0.5):
    """
    Apply progressive pinching that increases toward the end slices.
    """
    start_slice = int(total_slices * start_fraction)
    
    if slice_idx < start_slice:
        return mask
    
    normalized_pos = (slice_idx - start_slice) / (total_slices - start_slice - 1) if total_slices > start_slice + 1 else 1
    current_strength = normalized_pos * max_pinch_strength
    
    if pinch_type == 'radial':
        return radial_pinch(mask, current_strength)
    elif pinch_type == 'center_of_mass':
        return center_of_mass_pinch(mask, current_strength)
    elif pinch_type == 'distance':
        shrink_factor = 1.0 - current_strength
        return distance_based_shrink(mask, shrink_factor)
    else:
        return mask
    

def apply_progressive_shrinking(mask, slice_idx, total_slices, 
                               max_iterations=3, 
                               erosion_type='linear',
                               structure_type='cross',
                               start_fraction=0.5):
        """
        Apply progressive erosion to the entire mask (all non-zero values together).
        This preserves class boundaries while shrinking the overall segmentation.
        
        Args:
            mask: Multi-class mask (numpy array) with values 0, 1, 2, etc.
            slice_idx: Current slice index (0-based)
            total_slices: Total number of slices
            max_iterations: Maximum number of erosion iterations
            erosion_type: 'linear', 'quadratic', or 'exponential'
            structure_type: 'cross', 'square', or 'circle'
            start_fraction: Fraction of slices where erosion starts
        
        Returns:
            Shrunken mask with preserved class boundaries
        """
        # Calculate when to start erosion
        start_slice = int(total_slices * start_fraction)
        
        if slice_idx < start_slice:
            return mask  # No erosion for early slices
        
        # Calculate erosion strength based on slice position and type
        normalized_pos = (slice_idx - start_slice) / (total_slices - start_slice - 1) if total_slices > start_slice + 1 else 1
        
        if erosion_type == 'linear':
            erosion_factor = normalized_pos
        elif erosion_type == 'quadratic':
            erosion_factor = normalized_pos ** 2
        elif erosion_type == 'exponential':
            erosion_factor = (np.exp(normalized_pos * 2) - 1) / (np.exp(2) - 1)
        else:
            erosion_factor = normalized_pos
        
        erosion_iterations = int(erosion_factor * max_iterations)
        
        if erosion_iterations == 0:
            return mask
        
        # Define structuring element
        if structure_type == 'cross':
            structure = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=bool)
        elif structure_type == 'square':
            structure = np.ones((3, 3), dtype=bool)
        elif structure_type == 'circle':
            structure = generate_binary_structure(2, 1)
        else:
            structure = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=bool)
        
        # Create binary mask of ALL non-zero regions (classes 1, 2, etc.)
        binary_mask = mask > 0
        
        if not binary_mask.any():
            return mask  # No segmentation to erode
        
        # Apply erosion to the binary mask
        eroded_binary = binary_mask.copy()
        for _ in range(erosion_iterations):
            eroded_binary = binary_erosion(eroded_binary, structure=structure)
        
        # Apply the eroded binary mask to preserve only regions that survive erosion
        # This keeps the original class values but only where the eroded mask is True
        result_mask = np.zeros_like(mask)
        result_mask[eroded_binary] = mask[eroded_binary]
        
        return result_mask


#function that given a root directory, image pattern and label pattern, returns the median spacing of training images
def compute_spacing(root_dir, args, save=False):
    """
    Compute the median spacing of images in the dataset.
    Args:
        root_dir: Root directory of the dataset
        save: Whether to save the computed spacing to a file
    Returns:
        Median spacing as a tuple of floats
    """
    
    #check if spacing.txt already exists
    spacing_file = os.path.join(root_dir, "spacing.txt")
    if os.path.exists(spacing_file):
        print(f"Spacing file already exists: {spacing_file}")
        with open(spacing_file, "r") as f:
            spacing = f.read().strip().split()
            spacing = tuple(float(s) for s in spacing)
        print(f"Loaded spacing: {spacing}")
        return spacing
    
    paired_files = find_paired_files(
        root_dir=root_dir,
        image_pattern=args.image_pattern,
        label_pattern=args.label_pattern,
        recursive=True,
        case_sensitive=False
    )
    
    if not paired_files:
        raise ValueError(f"No paired files found in root dir: {root_dir} with patterns: "
                         f"images='{args.image_pattern}', labels='{args.label_pattern}'")
    print(f"Found {len(paired_files)} paired files")
    spacings = []
    for i, pair in enumerate(paired_files):
        #open with nibabel
        image_path = pair['label']
        import nibabel as nib
        img = nib.load(image_path)
        # orient image to RAS
        img = nib.as_closest_canonical(img)
        # get spacing
        spacing = img.header.get_zooms()[:3]
        if len(spacing) < 3:
            raise ValueError(f"Image {image_path} does not have 3D spacing information.")
        progress_bar(i + 1, len(paired_files), f"Spacing for {pair['label']}: {spacing}")
        spacings.append(spacing)
        
    median_spacing = np.median(spacings, axis=0)
    median_spacing = tuple(median_spacing)
    print(f"Computed median spacing: {median_spacing}\n")
    if save:
        with open(os.path.join(root_dir, "spacing.txt"), "w") as f:
            f.write(f"{median_spacing[0]} {median_spacing[1]} {median_spacing[2]}\n")
    return median_spacing


def create_transforms(args, channel=0, sample_axis=0):
    """
    Create the same transforms used during validation.
    
    Args:
        args: Configuration arguments
        channel: Channel to select for image
        sample_axis: Axis for sampling
    
    Returns:
        MONAI Compose transform
    """
    
    # Calculate size divisible and patch size
    size_divisible_3d = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    train_patch_size = args.autoencoder_train["patch_size"][:2]  # Take first 2 dimensions
    
    # Determine intensity scaling based on modality
    if hasattr(args, 'modality') and args.modality == 'mri':
        intensity_transform = ScaleIntensityRangePercentilesd(
            keys="image", lower=0, upper=99.5, b_min=0, b_max=1
        )
    elif hasattr(args, 'modality') and args.modality == 'ct':
        intensity_transform = ScaleIntensityRanged(
            keys="image", a_min=-57, a_max=164, b_min=0, b_max=1
        )
    elif hasattr(args, 'modality') and args.modality == 'us':
        intensity_transform = NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        )
    
    # Determine compute dtype
    compute_dtype = torch.float32
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        Lambdad(keys="label", func=lambda x: torch.where(x[0] > .5, 1, 0)) if args.num_classes == 1 else Lambdad(keys="label", func=lambda x: x[0]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropAroundNonZerod_Advanced(keys=["image", "label"], reference_key="label", axis=3),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        #DivisiblePadd(keys=["image", "label"], k=size_divisible_3d),
        #Resized(keys=["image", "label"], spatial_size=(train_patch_size[0], train_patch_size[1], -1), mode=("area", "nearest")),
        intensity_transform,
        CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
        Lambdad(keys=["slice_label"], func=lambda x: compute_slice_ratio_from_volume(x)[0, 0, 0]),
        Lambdad(keys=["image", "label"], func=lambda x: x.permute(3, 0, 1, 2)),
        EnsureTyped(keys=["image", "label", "slice_label"], dtype=compute_dtype),
    ])
    
    return val_transforms


def compute_slice_ratio_from_volume(volume):
    if volume.ndim > 3:
        volume = volume.squeeze(0)  # Remove channel
    if volume.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional")
    
    total_slices = volume.shape[-1]
    denom = max(total_slices - 1, 1)

    # normalized slice position in [0, 1] (0 = first slice, 1 = last slice).
    # Uses the slice index over total slices - not the foreground-slice count -
    # so the ratio is a clean, monotonic position signal even at apex/base.
    slice_ratios = torch.zeros(volume.shape, device=volume.device)
    for i in range(total_slices):
        slice_ratios[..., i] = i / denom
    return slice_ratios[None]


class CropAroundNonZerod_Advanced(Transform):
    """
    Advanced version with additional options for non-zero detection and cropping.
    
    Args:
        keys: Keys of the corresponding items to be transformed.
        reference_key: Key of the tensor used to determine non-zero regions.
        axis: The axis along which to find non-zero values and crop.
        margin: Additional margin to add around the non-zero region.
        threshold: Threshold for considering values as "non-zero" (default: 0).
        min_crop_size: Minimum size of the cropped region.
        allow_missing_keys: Don't raise exception if key is missing.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        reference_key: str,
        axis: int = -1,
        margin: int = 0,
        threshold: float = 0,
        min_crop_size: Optional[int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__()
        self.keys = keys
        self.reference_key = reference_key
        self.axis = axis
        self.margin = margin
        self.threshold = threshold
        self.min_crop_size = min_crop_size
        self.allow_missing_keys = allow_missing_keys
    
    def _find_nonzero_bounds(self, tensor: torch.Tensor, axis: int) -> tuple[int, int]:
        """Find bounds of values above threshold along the specified axis."""
        actual_axis = axis if axis >= 0 else tensor.ndim + axis
        
        if actual_axis >= tensor.ndim or actual_axis < 0:
            raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions")
        
        axes_to_sum = list(range(tensor.ndim))
        axes_to_sum.remove(actual_axis)
        
        if len(axes_to_sum) > 0:
            projection = torch.sum(torch.abs(tensor), dim=axes_to_sum)
        else:
            projection = torch.abs(tensor)
        
        above_threshold = projection > self.threshold
        nonzero_indices = torch.nonzero(above_threshold, as_tuple=True)[0]
        
        if len(nonzero_indices) == 0:
            return 0, tensor.shape[actual_axis]
        
        start_idx = nonzero_indices[0].item()
        end_idx = nonzero_indices[-1].item() + 1
        
        return start_idx, end_idx
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        if self.reference_key not in d:
            raise KeyError(f"Reference key '{self.reference_key}' not found in data")
        
        reference_tensor = d[self.reference_key]
        start_idx, end_idx = self._find_nonzero_bounds(reference_tensor, self.axis)
        
        actual_axis = self.axis if self.axis >= 0 else reference_tensor.ndim + self.axis
        axis_size = reference_tensor.shape[actual_axis]
        
        start_idx = max(0, start_idx - self.margin)
        end_idx = min(axis_size, end_idx + self.margin)
        
        if self.min_crop_size is not None:
            current_size = end_idx - start_idx
            if current_size < self.min_crop_size:
                additional_needed = self.min_crop_size - current_size
                expand_left = additional_needed // 2
                expand_right = additional_needed - expand_left
                
                start_idx = max(0, start_idx - expand_left)
                end_idx = min(axis_size, end_idx + expand_right)
                
                if end_idx - start_idx < self.min_crop_size:
                    if end_idx == axis_size:
                        start_idx = max(0, axis_size - self.min_crop_size)
                    elif start_idx == 0:
                        end_idx = min(axis_size, self.min_crop_size)
        
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in data")
            
            tensor = d[key]
            tensor_axis = self.axis if self.axis >= 0 else tensor.ndim + self.axis
            
            if tensor_axis >= tensor.ndim or tensor_axis < 0:
                raise ValueError(f"Axis {self.axis} is out of bounds for tensor '{key}'")
            
            if tensor.shape[tensor_axis] != reference_tensor.shape[actual_axis]:
                raise ValueError(f"Shape mismatch along axis {self.axis}")
            
            slices = [slice(None)] * tensor.ndim
            slices[tensor_axis] = slice(start_idx, end_idx)
            d[key] = tensor[tuple(slices)]
        
        return d


# --- File Finding Utilities ---

def extract_identifier(filename: str, pattern: str) -> str:
    """Extract identifier from filename based on pattern."""
    identifier = filename
    
    # Remove common extensions
    for ext in ['.nii.gz', '.nii', '.dcm', '.png', '.jpg', '.tiff']:
        if identifier.lower().endswith(ext):
            identifier = identifier[:-len(ext)]
            break
    
    # Remove pattern-specific parts
    pattern_parts = pattern.replace('*', '').replace('?', '').split('.')
    for part in pattern_parts:
        if part and part in identifier:
            identifier = identifier.replace(part, '')
    
    # Clean up common separators
    return identifier.strip('_-.')


def calculate_match_score(img_id: str, lbl_id: str, img_name: str, lbl_name: str) -> float:
    """Calculate how well two files match based on identifiers and names."""
    score = 0.0
    
    if img_id == lbl_id:
        score += 10.0  # Perfect ID match
    
    if img_id in lbl_id or lbl_id in img_id:
        score += 5.0  # Good subset match
    
    # Score based on common parts (e.g., 'train_1234' vs 'train_1234_0000')
    common_parts = set(img_id.split('_')) & set(lbl_id.split('_'))
    score += len(common_parts) * 2.0
    
    # Bonus for being in the same parent directory
    img_parts = img_name.split(os.sep)
    lbl_parts = lbl_name.split(os.sep)
    if len(img_parts) > 1 and len(lbl_parts) > 1:
        if img_parts[-2] == lbl_parts[-2]:
            score += 3.0
    return score


def find_paired_files(
    root_dir: str,
    image_pattern: str,
    label_pattern: str,
    recursive: bool = True,
    case_sensitive: bool = False
) -> List[Dict[str, str]]:
    """
    Find paired image and label files based on patterns.
    
    Args:
        root_dir: Root directory to search
        image_pattern: Glob pattern for image files
        label_pattern: Glob pattern for label files
        recursive: Whether to search recursively
        case_sensitive: Whether pattern matching is case sensitive
        
    Returns:
        List of dictionaries with 'image' and 'label' file paths
    """
    
    image_regex = re.compile(fnmatch.translate(image_pattern))
    label_regex = re.compile(fnmatch.translate(label_pattern))
    
    if not case_sensitive:
        # Re-compile regex with IGNORECASE flag
        image_regex = re.compile(fnmatch.translate(image_pattern), re.IGNORECASE)
        label_regex = re.compile(fnmatch.translate(label_pattern), re.IGNORECASE)
    
    search_pattern = "**/*" if recursive else "*"
    all_files = glob.glob(os.path.join(root_dir, search_pattern), recursive=recursive)
    
    image_files = []
    label_files = []
    
    print(f"Scanning {len(all_files)} files...")
    for file_path in all_files:
        if not os.path.isfile(file_path):
            continue
            
        filename = os.path.basename(file_path)
        
        # Use re.match with fnmatch.translate, which is a robust
        # way to match glob patterns against filenames.
        # The 'elif' is critical to prevent greedy matching.
        if image_regex.match(filename):
            image_files.append(file_path)
        elif label_regex.match(filename):
            label_files.append(file_path)
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels.")

    # --- CORRECT PAIRING LOGIC ---
    # This replaces the buggy list comprehension
    paired_files = []
    for img_path in sorted(image_files):
        img_name = os.path.basename(img_path)
        img_identifier = extract_identifier(img_name, image_pattern)
        
        best_match = None
        best_score = -1
        
        for lbl_path in sorted(label_files):
            lbl_name = os.path.basename(lbl_path)
            lbl_identifier = extract_identifier(lbl_name, label_pattern)
            
            score = calculate_match_score(img_identifier, lbl_identifier, img_name, lbl_name)
            
            if score > best_score:
                best_score = score
                best_match = lbl_path
        
        if best_match:
            paired_files.append({
                'image': img_path,
                'label': best_match
            })
    
    print(f"Successfully created {len(paired_files)} pairs.")
    return paired_files


def prepare_general_dataloader(
    args: str,
    image_pattern: str,
    label_pattern: str,
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sample_axis: int = 2,
    world_size: int = 1,
    cache: float = 0.0,
    randcrop: bool = False,
    size_divisible=4,
    num_center_slice=80,
    amp: bool = False,
    train_split: float = 0.8,
    validation_split: float = 0.2,
    test_split: float = 0.0,
    random_seed: int = 42,
    recursive: bool = True,
    case_sensitive: bool = False,
    **dataloader_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    
    print("Searching for paired files in: ", os.path.join("dataset", args.task))
    print(f"Image pattern: {image_pattern}")
    print(f"Label pattern: {label_pattern}")
    
    # Find paired files
    paired_files = find_paired_files(
        root_dir=os.path.join("dataset", args.task),
        image_pattern=image_pattern,
        label_pattern=label_pattern,
        recursive=recursive,
        case_sensitive=case_sensitive
    )
    
    if not paired_files:
        raise ValueError(f"No paired files found in given root dir with patterns: "
                        f"images='{image_pattern}', labels='{label_pattern}'")
    
    print(f"Found {len(paired_files)} paired files")
    
    # Validate splits
    assert abs(train_split + validation_split + test_split - 1.0) < 1e-6, \
        "train_split + validation_split + test_split must equal 1.0"
    
    # Split data
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(paired_files)).tolist()
    
    n_train = int(len(paired_files) * train_split)
    n_val = int(len(paired_files) * validation_split)
    n_test = len(paired_files) - n_train - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:] if n_test > 0 else []
    
    # Create datasets
    train_data = [paired_files[i] for i in train_indices]
    val_data = [paired_files[i] for i in val_indices]
    test_data = [paired_files[i] for i in test_indices] if test_indices else None
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data) if test_data else 0}")
    
    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    if sample_axis == 0:
        # sagittal
        train_patch_size = [1] + patch_size
        val_patch_size = [num_center_slice] + patch_size
        size_divisible_3d = [1, size_divisible, size_divisible]
    elif sample_axis == 1:
        # coronal
        train_patch_size = [patch_size[0], 1, patch_size[1]]
        val_patch_size = [patch_size[0], num_center_slice, patch_size[1]]
        size_divisible_3d = [size_divisible, 1, size_divisible]
    elif sample_axis == 2:
        # axial
        train_patch_size = patch_size + [1]
        val_patch_size = patch_size + [num_center_slice]
        size_divisible_3d = [size_divisible, size_divisible, 1]
    else:
        raise ValueError("sample_axis has to be in [0,1,2]")

    if randcrop:
        train_crop_transform = RandSpatialCropSamplesd(
            keys=["image", "label", "slice_label"],
            roi_size=train_patch_size,
            random_center=True,  # sample slices across the whole volume (incl. apex/base);
                                 # random_center=False collapsed every sample to the centre slice
            random_size=False,
            num_samples=batch_size,
        )
    else:
        train_crop_transform = CenterSpatialCropd(keys=["image", "label"], roi_size=val_patch_size)

    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if hasattr(args, 'modality') and args.modality == 'mri':
        intensity_transform = ScaleIntensityRangePercentilesd(
            keys="image", lower=0, upper=99.5, b_min=0, b_max=1
        )
    elif hasattr(args, 'modality') and args.modality == 'ct':
        intensity_transform = ScaleIntensityRanged(
            keys="image", a_min=-57, a_max=164, b_min=0, b_max=1
        )
    elif hasattr(args, 'modality') and args.modality == 'us':
        intensity_transform = NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        )    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            Lambdad(keys="label", func=lambda x: torch.where(x[0] > .5, 1, 0)) if args.num_classes == 1 else Lambdad(keys="label", func=lambda x:x[0]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            #Lambdad(keys=['image', 'label'], func=lambda x: print(x.shape)),
            #Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            CropAroundNonZerod_Advanced(keys=["image", "label"], reference_key="label", axis=3),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=(train_patch_size[0], train_patch_size[1], -1), mode=("area", "nearest")),
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            Lambdad(keys=["slice_label"], func=lambda x: compute_slice_ratio_from_volume(x)),
            #DivisiblePadd(keys=["image", "label"], k=size_divisible_3d),
            intensity_transform,
            train_crop_transform,
            SqueezeDimd(keys=["image", "label", "slice_label"], dim=1 + sample_axis),
            Lambdad(keys=["slice_label"], func=lambda x: x[0, 0, 0]),
            EnsureTyped(keys=["image", "label", "slice_label"], dtype=compute_dtype),
            
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            #Lambdad(keys='label', func=lambda x: print(x.shape)),
            Lambdad(keys="label", func=lambda x: torch.where(x[0] > .5, 1, 0)) if args.num_classes == 1 else Lambdad(keys="label", func=lambda x:x[0]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            #Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            CropAroundNonZerod_Advanced(keys=["image", "label"], reference_key="label", axis=3),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            #DivisiblePadd(keys=["image", "label"], k=size_divisible_3d),
            Resized(keys=["image", "label"], spatial_size=(train_patch_size[0], train_patch_size[1], batch_size), mode=("area", "nearest")),
            intensity_transform,
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            Lambdad(keys=["slice_label"], func=lambda x: compute_slice_ratio_from_volume(x)),
            SplitDimd(keys=["image", "label", "slice_label"], dim=1 + sample_axis, keepdim=False, list_output=True),
            Lambdad(keys=["slice_label"], func=lambda x: x[0, 0, 0]),
            EnsureTyped(keys=["image", "label", "slice_label"], dtype=compute_dtype),
        ]
    )
    
    # Create MONAI datasets
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=cache, num_workers=8)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=cache, num_workers=8)
    test_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=cache, num_workers=8) if test_data else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        **dataloader_kwargs
    )
    
    test_loader = None
    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            **dataloader_kwargs
        )
    
    # Print sample paths for verification
    print("\nSample paired files:")
    for i, pair in enumerate(train_data[:3]):
        print(f"  {i+1}. Image: {pair['image']}")
        print(f"     Label: {pair['label']}")
    
    print(f'TRAIN: Image shape {train_ds[0][0]["image"].shape}, Label shape {train_ds[0][0]["label"].shape}')
    print(f'VAL: Image shape {val_ds[0][0]["image"].shape}, Label shape {val_ds[0][0]["label"].shape}\n')
    
    if test_loader is None:
        return train_loader, val_loader
    
    return train_loader, val_loader, test_loader


class SelectRandomOrganChanneld(MapTransform):
    """
    Takes a multi-channel One-Hot Encoded (OHE) mask [C, H, W, D] and selects
    a single, non-background channel that contains a positive label.
    
    Returns a binary mask [1, H, W, D] for the selected organ.
    
    Args:
        keys: Keys of the OHE mask to transform.
        label_key: The key to store the selected class index (for conditioning).
        allow_background: If True, allows selecting the background 
                          channel (index 0) as a valid "organ".
    """
    def __init__(self, keys: KeysCollection, label_key: str = "label_class_index", allow_background: bool = False) -> None:
        super().__init__(keys)
        self.allow_background = allow_background
        self.label_key = label_key

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            ohe_mask = d[key] # Shape [C, H, W, D]
            
            # Determine which channels are present
            channel_sums = torch.sum(ohe_mask, dim=(1, 2, 3))
            
            if self.allow_background:
                present_channels = torch.where(channel_sums > 0)[0]
            else:
                # Only consider foreground (organ) channels [1:]
                present_channels = torch.where(channel_sums[1:] > 0)[0] + 1
            
            if len(present_channels) > 0:
                # Randomly select one
                selected_channel_idx = present_channels[
                    torch.randint(len(present_channels), (1,))
                ].item()
            else:
                # Fall back to background if no foreground organs are present
                selected_channel_idx = 0

            # Get the selected channel and add a channel dim
            # [C, H, W, D] -> [H, W, D] -> [1, H, W, D]
            d[key] = ohe_mask[selected_channel_idx].unsqueeze(0) 
            
            # Store the selected index for conditioning
            d[self.label_key] = torch.tensor(selected_channel_idx, dtype=torch.long)
            
        return d


def prepare_general_3d_dataloader(
    args: object,  # Use a simple object or Namespace for args
    image_pattern: str,
    label_pattern: str,
    num_classes_total: int, # Total number of classes for OHE (e.g., 15)
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    world_size: int = 1,
    cache_rate: float = 1.0,
    amp: bool = False,
    train_split: float = 0.8,
    validation_split: float = 0.2,
    test_split: float = 0.0,
    random_seed: int = 42,
    recursive: bool = True,
    case_sensitive: bool = False,
    num_workers: int = 4,
    **dataloader_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    
    print("Searching for paired files in: ", os.path.join("dataset", args.task))
    print(f"Image pattern: {image_pattern}")
    print(f"Label pattern: {label_pattern}")
    
    # 1. Find paired files (using your existing function)
    paired_files = find_paired_files(
        root_dir=os.path.join("dataset", args.task),
        image_pattern=image_pattern,
        label_pattern=label_pattern,
        recursive=recursive,
        case_sensitive=case_sensitive
    )
    
    if not paired_files:
        raise ValueError(f"No paired files found in given root dir with patterns: "
                        f"images='{image_pattern}', labels='{label_pattern}'")
    
    print(f"Found {len(paired_files)} paired files")    
    # 2. Split data
    assert abs(train_split + validation_split + test_split - 1.0) < 1e-6, \
        "train_split + validation_split + test_split must equal 1.0"
    
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(paired_files)).tolist()
    
    n_train = int(len(paired_files) * train_split)
    n_val = int(len(paired_files) * validation_split)
    n_test = len(paired_files) - n_train - n_val
    
    train_data = [paired_files[i] for i in indices[:n_train]]
    val_data = [paired_files[i] for i in indices[n_train:n_train + n_val]]
    test_data = [paired_files[i] for i in indices[n_train + n_val:]] if n_test > 0 else None
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data) if test_data else 0}")
    
    # 3. Setup DDP and Dtype
    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1, etc.
    compute_dtype = torch.float16 if amp else torch.float32

    # 4. Define Intensity Transform
    if hasattr(args, 'modality') and args.modality == 'mri':
        intensity_transform = ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )
    elif hasattr(args, 'modality') and args.modality == 'ct':
        intensity_transform = ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True # Example CT range
        )
    elif hasattr(args, 'modality') and args.modality == 'us':
        intensity_transform = NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        )    
    else: # Default to MRI
        intensity_transform = ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )
        
    # 5. Define 3D Transforms
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Select image channel (e.g., from multi-modal) and ensure label is Long
        Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"), # Add channel back
        Lambdad(keys="label", func=lambda x: x[0].long()), # Select first label channel, make Long
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"), # Add channel back
        
        Orientationd(keys=["image", "label"], axcodes="RAS", as_closest_canonical=True),
        Spacingd(
            keys=["image", "label"], 
            pixdim=spacing, 
            mode=("bilinear", "nearest")
        ),
        intensity_transform,
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        Resized(
            keys=["image", "label"], 
            spatial_size=(256, 256, 128), 
            mode=("trilinear", "nearest")
        ),
        AsDiscreted(keys="label", to_onehot=num_classes_total, dim=0),
        SelectRandomOrganChanneld(keys="label", label_key="label_class_index"),
        # Now 'label' is [1, H, W, D] binary mask for one organ
        # and 'label_class_index' holds the class ID (e.g., 3 for Liver)
        EnsureTyped(keys=["image", "label"], dtype=compute_dtype),
    ]
    
    # Define Train and Val transforms
    train_transforms = Compose(
        base_transforms + [
            RandSpatialCropd(
                keys=["image", "label"], 
                roi_size=patch_size, 
                random_size=False
            )
        ]
    )
    
    val_transforms = Compose(
        base_transforms + [
            RandSpatialCropd( # Validate on random patches as well
                keys=["image", "label"], 
                roi_size=patch_size, 
                random_size=False
            )
        ]
    )
    
    # 6. Create Datasets
    train_ds = CacheDataset(
        data=train_data, 
        transform=train_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    val_ds = CacheDataset(
        data=val_data, 
        transform=val_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    test_ds = CacheDataset(
        data=test_data, 
        transform=val_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    ) if test_data else None
    
    # 7. Create Dataloaders
    train_sampler = None
    val_sampler = None
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=torch.distributed.get_rank()
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=torch.distributed.get_rank(), shuffle=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(not ddp_bool),
        num_workers=num_workers,
        sampler=train_sampler,
        collate_fn=list_data_collate,
        pin_memory=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size, # Use batch_size=1 for validation if preferred
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        collate_fn=list_data_collate,
        pin_memory=True,
        **dataloader_kwargs
    )
    
    test_loader = None
    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size, # Use batch_size=1 for test
            shuffle=False,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
            **dataloader_kwargs
        )
    
    # Print sample paths for verification
    print("\nSample paired files (Train):")
    for i, pair in enumerate(train_data[:3]):
        print(f"  {i+1}. Image: {pair['image']}")
        print(f"     Label: {pair['label']}")
    
    return train_loader, val_loader, test_loader

class LoadNumpyd(MapTransform):
    """
    Dictionary-based transform to load a NumPy array from a file.
    """
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, any]) -> Dict[Hashable, any]:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in data.")

            file_path = d[key]
            if file_path is None:
                 raise FileNotFoundError(f"File path for key '{key}' was None.")

            try:
                np_data = np.load(file_path)
                d[key] = torch.from_numpy(np_data).float()
            except Exception as e:
                raise IOError(f"Could not load numpy file {file_path} for key '{key}'.") from e
        
        return d


class MapLabelsToTextd(MapTransform):
    """
    Dictionary-based transform to map a class index to a text string.
    """
    def __init__(
        self,
        keys: KeysCollection,
        label_map: Dict[int, str],
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        # Convert string keys from JSON to int
        self.label_map = {int(k): v for k, v in label_map.items()}

    def __call__(self, data: Mapping[Hashable, any]) -> Dict[Hashable, any]:
        d = dict(data)
        for key in self.keys:
            class_index = d[key].item() # Expects a 0-dim tensor
            if class_index not in self.label_map:
                # Fallback to background if class index is not in map
                print(f"Warning: Class index {class_index} not found in label_map. Defaulting to 'background'.")
                class_index = 0
                
            d[key + "_text"] = self.label_map[class_index]
        return d

def prepare_ldm_3d_dataloader(
    args: object,
    paired_files: List[Dict[str, str]],
    feature_dir: str, # Path to the pre-computed MAISI features
    num_classes_total: int,
    label_map: Dict[int, str], # Map from index to text
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    corruption_holes: Tuple[int, int] = (1, 8),
    corruption_spatial_size: Tuple[int, int] = (4, 16),
    world_size: int = 1,
    cache_rate: float = 1.0,
    amp: bool = False,
    train_split: float = 0.8,
    validation_split: float = 0.2,
    test_split: float = 0.0,
    random_seed: int = 42,
    num_workers: int = 8,
    **dataloader_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    
    if not paired_files:
        raise ValueError("No paired files provided.")
    
    # --- 1. Create new file list with feature paths ---
    ldm_file_list = []
    dataset_root = os.path.join("dataset", args.task)
    
    for pair in paired_files:
        image_path = pair["image"]
        label_path = pair["label"]
        
        # Get relative path of the image
        # e.g., "subject_001/train_001_0000.nii.gz"
        relative_path = os.path.relpath(image_path, start=dataset_root)
        
        # Change the extension to .npy
        if relative_path.endswith(".nii.gz"):
            feature_rel_path = relative_path[:-7] + ".npy"
        else:
            feature_rel_path = os.path.splitext(relative_path)[0] + ".npy"
            
        feature_path = os.path.join(feature_dir, feature_rel_path)
        
        if os.path.exists(feature_path):
            ldm_file_list.append({
                "label": label_path,
                "maisi_latent": feature_path
                # We don't need the original "image" path anymore
            })
        else:
            print(f"Warning: Skipping {image_path}, corresponding feature not found at {feature_path}")

    print(f"Found {len(ldm_file_list)} complete data triplets (label, feature).")

    # --- 2. Split data ---
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(ldm_file_list)).tolist()
    
    n_train = int(len(ldm_file_list) * train_split)
    n_val = int(len(ldm_file_list) * validation_split)
    n_test = len(ldm_file_list) - n_train - n_val
    
    train_data = [ldm_file_list[i] for i in indices[:n_train]]
    val_data = [ldm_file_list[i] for i in indices[n_train:n_train + n_val]]
    test_data = [ldm_file_list[i] for i in indices[n_train + n_val:]] if n_test > 0 else None
    
    print(f"LDM Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data) if test_data else 0}")
    
    ddp_bool = world_size > 1
    compute_dtype = torch.float16 if amp else torch.float32
        
    # --- 3. Define 3D Transforms ---
    
    # Keys for synchronous cropping
    crop_keys = ["label", "corrupted_label"]
    
    base_transforms = [
        # Load all 2 data types
        LoadImaged(keys=["label"]),
        LoadNumpyd(keys=["maisi_latent"]), # Loads .npy file
        
        EnsureChannelFirstd(keys=["label"]),
        
        # --- Prepare Label (for AE) ---
        Lambdad(keys="label", func=lambda x: x[0].long()),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["label"], axcodes="RAS", as_closest_canonical=True),
        AsDiscreted(keys="label", to_onehot=num_classes_total),
        SelectRandomOrganChanneld(keys="label", label_key="class_index"),
        MapLabelsToTextd(keys="class_index", label_map=label_map), # Adds "class_index_text"
        
        CopyItemsd(keys="label", times=1, names="corrupted_label"),
        RandCoarseDropoutd(
            keys="corrupted_label",
            holes=corruption_holes[0], # Min holes
            spatial_size=corruption_spatial_size,
            prob=1.0, # Always corrupt
            max_holes=corruption_holes[1], # Max holes
        ),
        
        EnsureTyped(keys=["label", "corrupted_label"], dtype=compute_dtype),
        # maisi_latent is already a float tensor
    ]
    
    # Note: We apply cropping *before* resampling (Spacingd)
    # This is much more efficient.
    
    train_transforms = Compose(
        base_transforms + [
            # 1. Crop Label and Corrupted Label in sync
            RandSpatialCropd(
                keys=crop_keys, 
                roi_size=patch_size, 
                random_size=False
            ),
            # 2. We don't need Spacingd here if the VAE already handled it.
            #    If the VAE was trained on patches *without* spacing,
            #    we should match that.
        ]
    )
    
    val_transforms = Compose(
        base_transforms + [
            # 1. Crop Label and Corrupted Label in sync
            RandSpatialCropd(
                keys=crop_keys, 
                roi_size=patch_size, 
                random_size=False
            )
        ]
    )
    
    # --- 4. Create Datasets ---
    train_ds = CacheDataset(
        data=train_data, 
        transform=train_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    val_ds = CacheDataset(
        data=val_data, 
        transform=val_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    
    # ... (rest of dataloader/sampler logic is unchanged) ...
    train_sampler = None
    val_sampler = None
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=torch.distributed.get_rank()
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=torch.distributed.get_rank(), shuffle=False
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers,
        sampler=train_sampler, collate_fn=list_data_collate, pin_memory=True, **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        sampler=val_sampler, collate_fn=list_data_collate, pin_memory=True, **dataloader_kwargs
    )
    
    test_loader = None # Not implemented here
    
    return train_loader, val_loader, test_loader


def prepare_msd_dataloader(
    args,
    batch_size,
    patch_size,
    spacing,
    amp=False,
    sample_axis=2,
    randcrop=True,
    rank=0,
    world_size=1,
    cache=0.0,
    download=False,
    size_divisible=4,
    num_center_slice=80,
):
    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    if sample_axis == 0:
        # sagittal
        train_patch_size = [1] + patch_size
        val_patch_size = [num_center_slice] + patch_size
        size_divisible_3d = [1, size_divisible, size_divisible]
    elif sample_axis == 1:
        # coronal
        train_patch_size = [patch_size[0], 1, patch_size[1]]
        val_patch_size = [patch_size[0], num_center_slice, patch_size[1]]
        size_divisible_3d = [size_divisible, 1, size_divisible]
    elif sample_axis == 2:
        # axial
        train_patch_size = patch_size + [1]
        val_patch_size = patch_size + [num_center_slice]
        size_divisible_3d = [size_divisible, size_divisible, 1]
    else:
        raise ValueError("sample_axis has to be in [0,1,2]")

    if randcrop:
        train_crop_transform = RandSpatialCropSamplesd(
            keys=["image", "label", "slice_label"],
            roi_size=train_patch_size,
            random_center=True,  # sample slices across the whole volume (incl. apex/base);
                                 # random_center=False collapsed every sample to the centre slice
            random_size=False,
            num_samples=batch_size,
        )
    else:
        train_crop_transform = CenterSpatialCropd(keys=["image", "label"], roi_size=val_patch_size)

    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            Lambdad(keys="label", func=lambda x: x[0, :, :, :]),
            #Lambdad(keys='label', func=lambda x: print(x.shape)),
            Lambdad(keys="label", func=lambda x: torch.where(x > .5, 1, 0)) if args.num_classes == 1 else Lambdad(keys="label", func=lambda x:x),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            #Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            CropAroundNonZerod_Advanced(keys=["image", "label"], reference_key="label", axis=3),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=(train_patch_size[0], train_patch_size[1], -1), mode=("area", "nearest")),
            #DivisiblePadd(keys=["image", "label"], k=size_divisible_3d),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.5, b_min=0, b_max=1
                ) if args.modality == 'mri' else ScaleIntensityRanged(
                keys="image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True
                ),
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            Lambdad(keys=["slice_label"], func=lambda x: compute_slice_ratio_from_volume(x)),
            train_crop_transform,
            SqueezeDimd(keys=["image", "label", "slice_label"], dim=1 + sample_axis),
            Lambdad(keys=["slice_label"], func=lambda x: x[0, 0, 0]),
            EnsureTyped(keys=["image", "label", "slice_label"], dtype=compute_dtype),
            
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            Lambdad(keys="label", func=lambda x: x[0, :, :, :]),
            #Lambdad(keys='label', func=lambda x: print(x.shape)),
            Lambdad(keys="label", func=lambda x: torch.where(x > .5, 1, 0)) if args.num_classes == 1 else Lambdad(keys="label", func=lambda x:x),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            #Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            CropAroundNonZerod_Advanced(keys=["image", "label"], reference_key="label", axis=3),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            #DivisiblePadd(keys=["image", "label"], k=size_divisible_3d),
            Resized(keys=["image", "label"], spatial_size=(train_patch_size[0], train_patch_size[1], -1), mode=("area", "nearest")),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=1
                ) if args.modality == 'mri' else ScaleIntensityRanged(
                keys="image", a_min=-57, a_max=164, b_min=0, b_max=1
                ),
            transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
            Lambdad(keys=["slice_label"], func=lambda x: compute_slice_ratio_from_volume(x)),
            SplitDimd(keys=["image", "label", "slice_label"], dim=1 + sample_axis, keepdim=False, list_output=True),
            Lambdad(keys=["slice_label"], func=lambda x: x[0, 0, 0]),
            EnsureTyped(keys=["image", "label", "slice_label"], dtype=compute_dtype),
        ]
    )
    
    os.makedirs(args.data_base_dir, exist_ok=True)
    train_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task=args.task,  # e.g., "Task01_BrainTumour"
        section="training",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task=args.task,  # e.g., "Task01_BrainTumour"
        section="validation",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=val_transforms,
    )
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler,
    )
    if rank == 0:
        print(f'TRAIN: Image shape {train_ds[0][0]["image"].shape}, Label shape {train_ds[0][0]["label"].shape}')
        print(f'VAL: Image shape {val_ds[0][0]["image"].shape}, Label shape {val_ds[0][0]["label"].shape}\n')
    
    return train_loader, val_loader


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


###############

def corrupt_ohe_masks(ohe_masks, 
                      corruption_prob=0.7,
                      fp_prob=0.3,
                      hole_prob=0.3, 
                      dilation_prob=0.2,
                      erosion_prob=0.2,
                      max_operations=2):
    """
    Apply random corruptions to one-hot encoded masks
    
    Args:
        ohe_masks: torch.Tensor of shape [B, C, H, W] where C is number of classes
        corruption_prob: Probability that a mask gets corrupted
        fp_prob: Probability of false positive augmentation
        hole_prob: Probability of hole augmentation  
        dilation_prob: Probability of dilation
        erosion_prob: Probability of erosion
        max_operations: Maximum number of operations to apply per mask
        
    Returns:
        torch.Tensor: Corrupted masks of same shape
    """
    device = ohe_masks.device
    dtype = ohe_masks.dtype
    batch_size, num_classes, height, width = ohe_masks.shape
    
    corrupted_masks = ohe_masks.clone()
    
    for b in range(batch_size):
        # Decide if this mask should be corrupted
        if random.random() > corruption_prob:
            continue
            
        # Decide which operations to apply
        operations = []
        if random.random() < fp_prob:
            operations.append('false_positive')
        if random.random() < hole_prob:
            operations.append('holes')
        if random.random() < erosion_prob:
            operations.append('erosion')
        if random.random() < dilation_prob:
            operations.append('dilation')
        
        # Limit number of operations
        if len(operations) > max_operations:
            operations = random.sample(operations, max_operations)
        
        # Apply selected operations
        current_mask = corrupted_masks[b].cpu().numpy()
        
        for operation in operations:
            if operation == 'false_positive':
                current_mask = _apply_false_positives(current_mask)
            elif operation == 'holes':
                current_mask = _apply_holes(current_mask)
            elif operation == 'dilation':
                current_mask = _apply_dilation(current_mask)
            elif operation == 'erosion':
                current_mask = _apply_erosion(current_mask)
        
        corrupted_masks[b] = torch.from_numpy(current_mask).to(device).to(dtype)
    
    return corrupted_masks

def _apply_false_positives(mask):
    """
    Add false positives by mirroring and copying random portions
    Args:
        mask: numpy array of shape [C, H, W]
    Returns:
        numpy array with false positives added
    """
    num_classes, height, width = mask.shape
    
    # Skip background channel (index 0), work on foreground channels
    for c in range(0, num_classes):
        if not np.any(mask[c] > 0.5):
            continue  # Skip if channel is empty
            
        # Find existing regions in this channel
        existing_regions = mask[c] > 0.5
        
        if not np.any(existing_regions):
            continue
        
        # Get bounding box of existing regions
        coords = np.where(existing_regions)
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract a random portion of the existing region
        region_height = y_max - y_min + 1
        region_width = x_max - x_min + 1
        
        # Random crop size (20-80% of original region)
        crop_ratio = random.uniform(0.2, 0.8)
        crop_h = max(1, int(region_height * crop_ratio))
        crop_w = max(1, int(region_width * crop_ratio))
        
        # Random crop position within the region
        crop_y = random.randint(y_min, max(y_min, y_max - crop_h + 1))
        crop_x = random.randint(x_min, max(x_min, x_max - crop_w + 1))
        
        # Extract the crop
        crop = mask[c, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Find a new location to place the false positive
        # Try to place it away from existing regions
        distance_map = ndimage.distance_transform_edt(~existing_regions)
        
        # Find suitable locations (far enough from existing regions)
        suitable_locations = distance_map > 5  # At least 5 pixels away
        
        if np.any(suitable_locations):
            # Get possible top-left corners for placement
            possible_y, possible_x = np.where(suitable_locations)
            
            # Filter positions where the crop would fit
            valid_positions = []
            for i in range(len(possible_y)):
                y, x = possible_y[i], possible_x[i]
                if (y + crop_h <= height) and (x + crop_w <= width):
                    valid_positions.append((y, x))
            
            if valid_positions:
                # Randomly choose a position
                new_y, new_x = random.choice(valid_positions)
                
                # Apply mirroring (flip horizontally or vertically randomly)
                if random.random() < 0.5:
                    crop = np.fliplr(crop)
                if random.random() < 0.5:
                    crop = np.flipud(crop)
                
                # Place the false positive
                mask[c, new_y:new_y+crop_h, new_x:new_x+crop_w] = np.maximum(
                    mask[c, new_y:new_y+crop_h, new_x:new_x+crop_w],
                    crop
                )
    
    return mask

def _apply_holes(mask):
    """
    Create holes in existing regions
    Args:
        mask: numpy array of shape [C, H, W]
    Returns:
        numpy array with holes added
    """
    num_classes, height, width = mask.shape
    
    for c in range(0, num_classes):  # Skip background
        if not np.any(mask[c] > 0.5):
            continue
            
        existing_regions = mask[c] > 0.5
        
        # Find connected components
        labeled, num_features = ndimage.label(existing_regions)
        
        for region_id in range(1, num_features + 1):
            region_mask = (labeled == region_id)
            region_size = np.sum(region_mask)
            
            # Only create holes in reasonably sized regions
            if region_size < 100:  # Too small for holes
                continue
                
            # Number of holes to create (1-3)
            num_holes = random.randint(1, 3)
            
            for _ in range(num_holes):
                # Find region coordinates
                coords = np.where(region_mask)
                if len(coords[0]) == 0:
                    continue
                
                # Random center for hole
                idx = random.randint(0, len(coords[0]) - 1)
                center_y, center_x = coords[0][idx], coords[1][idx]
                
                # Random hole size (5-15% of region size)
                hole_size = random.randint(
                    max(2, int(np.sqrt(region_size) * 0.1)),
                    max(3, int(np.sqrt(region_size) * 0.3))
                )
                
                # Create circular hole
                y, x = np.ogrid[:height, :width]
                hole_mask = (y - center_y)**2 + (x - center_x)**2 <= hole_size**2
                
                # Apply hole only within the current region
                hole_mask = hole_mask & region_mask
                mask[c][hole_mask] = 0.0
    
    return mask

def _apply_dilation(mask):
    """
    Apply dilation to regions
    Args:
        mask: numpy array of shape [C, H, W]
    Returns:
        numpy array with dilated regions
    """
    num_classes, height, width = mask.shape
    
    for c in range(0, num_classes):  # Skip background
        if not np.any(mask[c] > 0.5):
            continue
            
        # Random dilation iterations (1-3)
        iterations = random.randint(1, 3)
        
        # Apply dilation
        dilated = ndimage.binary_dilation(
            mask[c] > 0.5, 
            iterations=iterations
        ).astype(np.float32)
        
        mask[c] = dilated
    
    return mask

def _apply_erosion(mask):
    """
    Apply erosion to regions
    Args:
        mask: numpy array of shape [C, H, W]
    Returns:
        numpy array with eroded regions
    """
    num_classes, height, width = mask.shape
    
    for c in range(0, num_classes):  # Skip background
        if not np.any(mask[c] > 0.5):
            continue
            
        # Random erosion iterations (1-2, less aggressive than dilation)
        iterations = random.randint(2, 4)
        
        # Apply erosion
        eroded = ndimage.binary_erosion(
            mask[c] > 0.5, 
            iterations=iterations
        ).astype(np.float32)
        
        mask[c] = eroded
    
    return mask

# Utility function for visualization
def visualize_corruption_comparison(original, corrupted, save_path=None):
    """
    Visualize original vs corrupted masks side by side
    """
    import matplotlib.pyplot as plt
    
    # Convert to class indices for visualization
    orig_classes = torch.argmax(original, dim=1)[0].cpu().numpy()
    corr_classes = torch.argmax(corrupted, dim=1)[0].cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(orig_classes, cmap='tab10')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(corr_classes, cmap='tab10')
    ax2.set_title('Corrupted')
    ax2.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    
class CustomAE(nn.Module):
    def __init__(self, latent_size=32, classes=1):
        super().__init__()
        self.init_layers(latent_size, classes)
        self.apply(self.weight_init)

    def init_layers(self, latent_size, classes):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=classes, out_channels=32, kernel_size=4, stride=2, padding=1),
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

            nn.ConvTranspose2d(in_channels=32, out_channels=classes, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class MultiComponentLoss():
    def __init__(self, functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss):
        self.MSELoss = self.MSELoss()
        self.BKMSELoss = self.BKMSELoss()
        self.BKGDLoss = self.BKGDLoss()
        self.GDLoss = self.GDLoss()
        self.functions = functions
        self.settling_epochs_BKGDLoss = settling_epochs_BKGDLoss
        self.settling_epochs_BKMSELoss = settling_epochs_BKMSELoss

    class BKMSELoss:
        def __init__(self):
            self.MSELoss = nn.MSELoss()
        def __call__(self, prediction, target):
            return self.MSELoss(prediction, target)

    class MSELoss:
        def __init__(self):
            self.MSELoss = nn.MSELoss()
        def __call__(self, prediction, target):
            return self.MSELoss(prediction[:,1:], target[:,1:])

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

    def __call__(self, prediction, target, epoch, validation=False):
        contributes = {f: self.__dict__[f](prediction, target) for f in self.functions}
        if "BKGDLoss" in contributes and epoch < self.settling_epochs_BKGDLoss:
            contributes["BKGDLoss"] += self.BKGDLoss(prediction[:,1:], target[:,1:])
        if "BKMSELoss" in contributes and epoch < self.settling_epochs_BKMSELoss:
            contributes["BKMSELoss"] += self.BKMSELoss(prediction[:,1:], target[:,1:])
        contributes["Total"] = sum(contributes.values())
        if validation:
            return {k: v.item() for k,v in contributes.items()}
        else:
            return contributes["Total"]

