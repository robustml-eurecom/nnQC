import os
import numpy as np
import torch
from batchgenerators.augmentations.utils import resize_segmentation

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
    SpatialPadd,
    AsDiscreted,
) 
import monai
from monai.data import DataLoader
from glob import glob
import random

def check_data(ds):
    check_loader = DataLoader(ds, batch_size=1)
    check_data = monai.utils.misc.first(check_loader)
    print("first volume's shape: ", check_data["img"].shape, check_data["seg"].shape)
    

def get_dataloader(
    root_dir, 
    mode, 
    keyword, 
    batch_size, 
    classes, 
    transforms, 
    sanity_check=False
    ):
    img_dir = os.path.join(root_dir, "img_training")
    mask_dir = os.path.join(root_dir, "training")
    images = sorted(glob(os.path.join(img_dir, f"*{keyword}*")))
    segs = sorted(glob(os.path.join(mask_dir, f"*{keyword}*")))
    
    total = len(images)
    if mode=="train":
        images = images[:int(0.8*total)]
        segs = segs[:int(0.8*total)]
    elif mode=="val":
        images = images[int(0.8*total):int(0.9*total)]
        segs = segs[int(0.8*total):int(0.9*total)]
    elif mode=="test":
        images = images[int(0.9*total):]
        segs = segs[int(0.9*total):]
    
    data = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    
    transforms = get_transforms(mode, classes=classes)
    
    
    volume_ds = monai.data.CacheDataset(data=data, transform=transforms)
    
    patch_func = monai.data.PatchIterd(
        keys=["img", "seg"],
        patch_size=(1, None, None),  # dynamic last two dimensions
        start_pos=(0, 0, 0)
    )
    patch_transform = get_transforms("patch", classes=classes)
    
    patch_ds = monai.data.GridPatchDataset(
        data=volume_ds, patch_iter=patch_func, 
        transform=patch_transform, with_coordinates=False
    )
    
    if sanity_check:
        check_data(patch_ds)
        
    return DataLoader(
        patch_ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        shuffle=False
    )


class Resizer(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_segmentation(self, image, new_shape):
        return resize_segmentation(image, new_shape, order=1)
  
    def __call__(self, sample):
        sample = self.resize_segmentation(sample, new_shape=self.output_size)
        return sample


class AddPadding(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_by_padding(self, image, new_shape, pad_value=0):
        shape = tuple(list(image.shape))
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
        if pad_value is None:
            if len(shape) == 2:
                pad_value = image[0, 0]
            elif len(shape) == 3:
                pad_value = image[0, 0, 0]
            else:
                raise ValueError("Image must be either 2 or 3 dimensional")
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2. - np.array(shape) / 2.
        if len(shape) == 2:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
        elif len(shape) == 3:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
            int(start[2]):int(start[2]) + int(shape[2])] = image
        return res
  
    def __call__(self, sample):
        sample = self.resize_image_by_padding(sample, new_shape=self.output_size)
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self, img, crop_size):
        if all(np.array(img.shape) <= crop_size):
            return img
        center = np.array(img.shape) / 2.
        if type(crop_size) not in (tuple, list):
            center_crop = [int(crop_size)] * len(img.shape)
        else:
            center_crop = crop_size
            assert len(center_crop) == len(img.shape)
        return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

    def __call__(self, sample):
        sample = self.center_crop_2D_image(sample, crop_size=self.output_size)
        return sample


class OneHot(object):
    def __init__(self, num_classes=4, background=True):
        self.num_classes = num_classes
    def one_hot(self, seg):
        return np.eye(self.num_classes)[seg.astype(int)].transpose(2,0,1)
    def __call__(self, sample):
        sample = self.one_hot(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample


class MirrorTransform():
    def __call__(self, sample):
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[::-1])
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[:, ::-1])
        return sample


class Patient(torch.utils.data.Dataset):
    def __init__(self, root_dir, patient_id, transform=None, 
                 condition=False, multi=False, forceToOne=False, isimg=False):
        self.root_dir = root_dir
        self.id = patient_id
        self.info = np.load(os.path.join(('/').join(self.root_dir.split('/')[:-1]), "patient_info.npy"), allow_pickle=True).item()[patient_id]
        self.transform = transform
        self.condition = condition
        self.multi = multi
        self.forceToOne = forceToOne
        self.isimg = isimg
        
    def __len__(self):
        return (self.info['crop'][-1].stop - self.info['crop'][-1].start)

    def __getitem__(self, slice_id):
        data = np.load(os.path.join(self.root_dir, "patient{:04d}.npy".format(self.id)))
        sample = data[slice_id]
        #print(sample.shape, self.__len__(), slice_id)
        if self.transform:
            if self.forceToOne:
                sample = np.where(sample!=0, 1, 0)
            if self.isimg:
                sample = sample[None,:,:]
                sample = self.transform(sample)
                sample = sample[0].numpy()
                sample = AddPadding((256, 256))(sample)
                sample = CenterCrop((256, 256))(sample)
                sample = sample[None,:,:]    
            else:
                sample = self.transform(sample) 
            
        if self.condition:
            return sample, slice_id/self.__len__()
        
        if self.multi:
            pad_slices = np.zeros_like(sample)
            if slice_id-1 < 0:
                next_slice = data[slice_id+1]
                next_slice = self.transform(next_slice)
                sample = np.concatenate([pad_slices, sample, next_slice], axis=0)
            elif slice_id+1 >= self.__len__(): 
                prev_slice = data[slice_id-1]
                prev_slice = self.transform(prev_slice)
                sample = np.concatenate([prev_slice, sample, pad_slices], axis=0)
            else:
                prev_slice = data[slice_id-1]
                prev_slice = self.transform(prev_slice)
                next_slice = data[slice_id+1]
                next_slice = self.transform(next_slice)
                sample = np.concatenate([prev_slice, sample, next_slice], axis=0)
        return sample

class TestDataLoader():
    def __init__(self, root_dir, patient_ids, batch_size=None, 
                 transform=None, condition=False, multi=False, forceToOne=False, isimg=False):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.transform = transform
        self.condition = condition
        self.multi = multi
        self.forceToOne = forceToOne
        self.patient_loaders = []
        if batch_size is not None:
            for id in self.patient_ids:
                self.patient_loaders.append(torch.utils.data.DataLoader(
                    Patient(root_dir, id, transform=transform, 
                            condition=condition, multi=self.multi, forceToOne=self.forceToOne, isimg=isimg),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ))
        self.counter_id = 0
    
    def set_batch_size(self, batch_size):
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(torch.utils.data.DataLoader(
                Patient(self.root_dir, id, transform=self.transform, 
                            condition=self.condition, multi=self.multi),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))
    
    def set_transform(self, transform):
        self.transform = transform
        for loader in self.patient_loaders:
            loader.dataset.transform = transform

    def __iter__(self):
        self.counter_iter = 0
        return self

    def __next__(self):
        if(self.counter_iter == len(self)):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id%len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]


def get_transforms(mode, classes):
    if mode == "train":
        return Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys="img"),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    elif mode == "val" or mode == "test":
        return Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys="img"),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    elif mode == "patch":
        return Compose(
            [
                SqueezeDimd(keys=["img", "seg"], dim=0),  # squeeze the last dim
                Resized(keys=["img", "seg"], spatial_size=[128, 128]),
                SpatialPadd(keys=["img", "seg"], spatial_size=[256, 256]),
                AsDiscreted(keys=["seg"], to_onehot=classes),
            ]
        )

