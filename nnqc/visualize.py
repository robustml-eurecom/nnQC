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

import numpy as np
from monai.utils.type_conversion import convert_to_numpy


def normalize_image_to_uint8(mask):
   if mask.size == 0:
       return np.zeros_like(mask, dtype=np.uint8)
   
   unique_values = np.unique(mask)
   
   if len(unique_values) <= 2:
       # Binary mask
       result = np.zeros_like(mask, dtype=np.uint8)
       result[mask != 0] = 255
       return result
   else:
       # Multiclass mask
       mask_normalized = mask.copy().astype(np.float32)
       mask_normalized = (mask_normalized / np.max(mask_normalized) * 255).astype(np.uint8)
       return mask_normalized


def visualize_2d_image(image):
    """
    Prepare a 2D image for visualization.
    Args:
        image: image numpy array, sized (H, W)
    """
    image = convert_to_numpy(image)
    # draw image
    draw_img = normalize_image_to_uint8(image)
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img
