import os
import numpy as np
import nibabel as nib
import shutil
import gzip
from tqdm import tqdm
import SimpleITK as sitk
import glob

from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation, resize


def crop_image(image):
    nonzero_mask = binary_fill_holes(image != 0)
    mask_voxel_coords = np.stack(np.where(nonzero_mask))
    minidx = np.min(mask_voxel_coords, axis=1)
    maxidx = np.max(mask_voxel_coords, axis=1) + 1
    resizer = tuple([slice(*i) for i in zip(minidx,maxidx)])
    return resizer

def compute_spacing(spacing):
    if len(spacing) == 4:
        # Combine the spacing of the last two dimensions
        combined_spacing = spacing[:2] + (spacing[2]**2 + spacing[3]**2)**.5,
        return combined_spacing
    return spacing

def process_mask(mask):
    mask = mask.get_fdata()
    if len(mask.shape) == 4:
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)
        non_zero_slices = np.any(mask != 0, axis=(0, 1))
        mask = mask[:, :, non_zero_slices]
        #print(mask.shape)
        return mask, non_zero_slices
    return mask, []

def generate_patient_info(folder, patient_ids, start_idx=0):
    num_patients = patient_ids.stop - patient_ids.start
    patient_info = {}
    missing = 0
    progress_bar = tqdm(patient_ids, desc="Generating patient info")
    for id in progress_bar:
        progress_bar.set_postfix_str(f"Patient {id}. Missing: {missing}/{num_patients}")
        patient_folder = os.path.join(folder, 'patient{:04d}'.format(id))
        mask_path = os.path.join(patient_folder, "mask.nii.gz")
        image_path = glob.glob(os.path.join(patient_folder, "image*"))[0]
        
        if not os.path.exists(mask_path) or not os.path.exists(image_path):
            tqdm.write(f"Patient {id} is missing mask or image file")
            missing += 1
            continue
        
        mask = nib.load(mask_path)
        if 'nii' in image_path:
            image = nib.load(image_path)
            header = image.header   
            spacing = header.get_zooms()
            affine = image.affine
            image = image.get_fdata()
        elif 'mha' in image_path:
            image = sitk.ReadImage(image_path)
            header = {k:image.GetMetaData(k) for k in image.GetMetaDataKeys()}
            spacing = image.GetSpacing()
            affine = image.GetOrigin()
            image = sitk.GetArrayFromImage(image)
        
        mask, non_zero_slices = process_mask(mask)
            
        if np.unique(mask).shape[0] == 1:
            #tqdm.write(f"Patient {id} has no mask")
            missing += 1
            continue
        
        patient_info[id+start_idx] = {'ID': id}
        patient_info[id+start_idx]['classes'] = np.unique(mask).shape[0]
        patient_info[id+start_idx]["shape"] = image.shape
        patient_info[id+start_idx]["crop"] = crop_image(mask)
        patient_info[id+start_idx]["spacing"] = spacing[:3]
        patient_info[id+start_idx]["header"] = header
        patient_info[id+start_idx]["affine"] = affine
        patient_info[id+start_idx]["non_zero_slices"] = non_zero_slices
    return patient_info
  
  
def preprocess_image(image, crop, spacing, spacing_target, raw=False, mha=False):
    image = image[crop].transpose(2,1,0) if not mha else image.transpose(2,1,0)[crop].transpose(2,1,0)
    spacing_target[0] = spacing[0]
    new_shape = np.round(spacing / spacing_target * image.shape).astype(int)
    image = resize_segmentation(image, new_shape, order=1) if not raw else resize(image, new_shape, order=1)
    return image


def preprocess(patient_ids, start_idx, patient_info, spacing_target, folder, folder_out, get_patient_folder, get_fname, raw):
    progress_bar = tqdm(patient_ids, desc="Preprocessing")
    missing = 0
    mha = False
    for id in progress_bar:
        progress_bar.set_postfix_str(f"Patient {id}. Missing: {missing}/{len(patient_ids)}")
        if id not in patient_info:
            missing += 1
            tqdm.write(f"Patient {id} is missing info")
            continue
        
        patient_folder = get_patient_folder(folder, id)
        images = []
        fname = get_fname(patient_info, id)
        fname = os.path.join(patient_folder, fname)

        if(not os.path.isfile(fname)):
            continue
        
        if 'nii' in fname:
            image = nib.load(fname)
            image_arr = image.get_fdata()
        elif 'mha' in fname:
            image = sitk.ReadImage(fname)
            image_arr = sitk.GetArrayFromImage(image)
            mha = True
        
        if len(patient_info[id+start_idx]["non_zero_slices"]) > 0:
            image_arr = image_arr.reshape(image_arr.shape[0], image_arr.shape[1], -1)
            image_arr = image_arr[:,:,patient_info[id+start_idx]["non_zero_slices"]]
            
        image = preprocess_image(
            image_arr,
            patient_info[id+start_idx]["crop"],
            patient_info[id+start_idx]["spacing"],
            spacing_target,
            raw=raw,
            mha=mha
        )
        images.append(image)
        images = np.vstack(images)
        np.save(os.path.join(folder_out, "patient{:04d}".format(id+start_idx)), images.astype(np.float32))
        

def apply_preprocessing(
    patient_ids, start_idx, patient_info, spacing_target, folder, folder_out, 
    get_patient_folder, get_fname, raw=False
    ):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    preprocess(
        patient_ids, 
        start_idx,
        patient_info, 
        spacing_target,
        folder, 
        folder_out, 
        get_patient_folder, 
        get_fname, 
        raw
    )
    

def gunzip_and_replace(filePath: str):
    with open(filePath, "rb") as f_in:
        with gzip.open(filePath + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
            f_out.close()
        f_in.close()
    os.remove(filePath)


def find_segmentations(root_dir: os.PathLike, keywords: list, absolute: bool = False) -> list:
    assert type(keywords) != str, "Parameter keywords must be a list of str."
    segmentations = [[]]
    cwd = os.getcwd() if absolute else ""
    for dirElement in os.listdir(root_dir):
        subPath = os.path.join(root_dir, dirElement)
        if os.path.isdir(subPath):
            segmentations.append(find_segmentations(subPath, keywords, absolute))
        else:
            for keyword in keywords:
                if keyword in subPath:
                    path = os.path.join(cwd, subPath)
                    segmentations.append(path)

    return np.unique(np.hstack(segmentations))


def find_non_matching_pairs(mask_paths, image_paths):
    mask_set = set([os.path.dirname(path) for path in mask_paths])
    image_set = set([os.path.dirname(path) for path in image_paths])
    missing_files = mask_set.symmetric_difference(image_set)
    missing_mask_pairs = [p for p in mask_paths if os.path.dirname(p) in missing_files and "mask.nii.gz" in p]
    missing_image_pairs = [p for p in image_paths if os.path.dirname(p) in missing_files and "image.nii.gz" in p]

    return missing_mask_pairs, missing_image_pairs


def ignore_directories(mask_paths, image_paths):
    missing_mask_pairs, missing_image_pairs = find_non_matching_pairs(mask_paths, image_paths)
    missing_image_pairs = [os.path.dirname(p) for p in missing_image_pairs]
    missing_mask_pairs = [os.path.dirname(p) for p in missing_mask_pairs]

    ignore_directories = np.unique(missing_mask_pairs, missing_image_pairs)

    return ignore_directories


def find_pairs(root_dir: os.PathLike, mask_keywords, image_keywords, absolute: bool = False):
    mask_paths = find_segmentations(root_dir=root_dir, keywords=mask_keywords, absolute=absolute)
    image_paths = find_segmentations(root_dir=root_dir, keywords=image_keywords, absolute=absolute)
    ignore = ignore_directories(mask_paths, image_paths)
    filtered_masks = []
    for p in mask_paths:
        if not any([i in p for i in ignore]):
            filtered_masks.append(p)
    filtered_images = []
    for p in image_paths:
        if not any([i in p for i in ignore]):
            filtered_images.append(p)

    return filtered_masks, filtered_images


def structure_dataset(
    data_path: str,
    mask_paths: os.PathLike,
    image_paths: os.PathLike = None,
    destination_folder: str = "structured",
    maskName: str = "mask.nii.gz",
    imageName: str = "image.nii.gz",
) -> None:

    for path in mask_paths:
        assert path.endswith("nii.gz") or path.endswith(
            ".nii"
        ), f"segmentation must be of type .nii or .nii.gz but {path} was given."
    if image_paths is not None:
        for path in image_paths:
            assert path.endswith("nii.gz") or path.endswith(
                ".nii"
            ), f"image must be of type .nii or .nii.gz but {path} was given."

    assert maskName.endswith(".nii.gz"), "`maskName` must end with .nii.gz"
    assert imageName.endswith(".nii.gz"), "`fileName` must end with .nii.gz"

    destination_folder = os.path.join(data_path, destination_folder)

    os.makedirs(destination_folder) if not os.path.exists(destination_folder) else None

    mask_paths.sort()
    destination_folder = os.path.join(os.getcwd(), destination_folder)
    os.makedirs(destination_folder) if not os.path.exists(destination_folder) else None

    for i in range(len(mask_paths)):
        mask_path = mask_paths[i]
        image_path = image_paths[i] if image_paths is not None else None
        convert_mask = True if mask_path.endswith(".nii") else False
        if image_path is not None:
            convert_image = True if image_path.endswith(".nii") else False
        else:
            convert_image = False

        patient_target_folder = os.path.join(destination_folder, "patient{:04d}".format(i))
        os.makedirs(patient_target_folder) if not os.path.exists(patient_target_folder) else None

        target_name_mask = (
            os.path.join(patient_target_folder, maskName[:-3])
            if convert_mask
            else os.path.join(patient_target_folder, maskName)
        )
        target_name_image = (
            os.path.join(patient_target_folder, imageName[:-3])
            if convert_image
            else os.path.join(patient_target_folder, imageName)
        )

        os.rename(mask_path, target_name_mask)
        if convert_mask:
            gunzip_and_replace(target_name_mask)

        if image_paths is not None:
            os.rename(image_path, target_name_image)
        if convert_image:
            gunzip_and_replace(target_name_image)

