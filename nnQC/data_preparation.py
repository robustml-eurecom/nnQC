import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from glob import glob

from utils.preprocess import find_segmentations, structure_dataset


def get_image_seg_pairs(keywords, root_dir, absolute=False):
    total_paths = glob(os.path.join(root_dir, "**", "*"), recursive=True)
    image_paths = find_segmentations(root_dir, keywords["image"], absolute)
    # remove image_paths from total_paths
    total_paths = [path for path in total_paths if path not in image_paths and path.endswith(".nii.gz")]
    if keywords["seg"] is None:
        mask_paths = sorted(total_paths)
    else:
        mask_paths = find_segmentations(root_dir, keywords["seg"], absolute)
    return image_paths, mask_paths


def format_data(image_paths, mask_paths, destination_folder, startIdx, maskName, imageName):
    for i, (img, mask) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths), desc="Formatting data", postfix="Subject"):
        os.makedirs(os.path.join(destination_folder, f"patient{i+startIdx:04d}"), exist_ok=True)
        shutil.copyfileobj(open(img, "rb"), open(os.path.join(destination_folder, f"patient{i+startIdx:04d}", imageName), "wb"))
        shutil.copyfileobj(open(mask, "rb"), open(os.path.join(destination_folder, f"patient{i+startIdx:04d}", maskName), "wb"))
        

def process_one_class(path, use_class):
    mask = nib.load(path)
    affine = mask.affine
    mask = mask.get_fdata()
    assert use_class > 0, "Class must be greater than 0. The class 0 is reserved for background only."
    mask[mask != use_class] = 0
    mask[mask == use_class] = 1
    mask = nib.Nifti1Image(mask, affine)
    return mask


def split_training_testing(data_folder, split=0.8, use_class=-1):
    patient_list = os.listdir(data_folder)
    train_idx = int(len(patient_list) * split)
    train_patients = patient_list[:train_idx]
    test_patients = patient_list[train_idx:]
    
    root = data_folder.split('/')[0]
    train_folder = os.path.join(root, "training")
    test_folder = os.path.join(root, "testing")
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    for patient in train_patients:
        if use_class != -1:
            mask = process_one_class(os.path.join(data_folder, patient, "mask.nii.gz"), use_class)
            nib.save(mask, os.path.join(data_folder, patient, "mask.nii.gz"))
        shutil.move(os.path.join(data_folder, patient), os.path.join(train_folder, patient))
    print(f"Training data saved to {train_folder}")
    for patient in test_patients:  
        if use_class != -1: 
            mask = process_one_class(os.path.join(data_folder, patient, "mask.nii.gz"), use_class)
            nib.save(mask, os.path.join(data_folder, patient, "mask.nii.gz"))
        shutil.move(os.path.join(data_folder, patient), os.path.join(test_folder, patient))
    print(f"Testing data saved to {test_folder}")
    
    os.removedirs(data_folder)


def run(args):
    keywords = {
        "image": args.image_keywords,
        "seg": args.mask_keywords
    }
    image_paths, mask_paths = get_image_seg_pairs(keywords, args.root_dir)
    print(f"Found {len(image_paths)} image files and {len(mask_paths)} mask files.")
    print(f"First image file: {image_paths[0]}")
    print(f"First mask file: {mask_paths[0]}")

    if args.destination_folder is not None:
        if not os.path.exists(args.destination_folder):
            os.makedirs(args.destination_folder)
        format_data(
            image_paths, 
            mask_paths, 
            args.destination_folder, 
            args.startIdx,
            args.maskName, 
            args.imageName
        )
        print(f"Structured temporary dataset saved to {args.destination_folder}")

    if args.save_csv:
        df = pd.DataFrame({
            "image": image_paths,
            "mask": mask_paths
        })
        df.to_csv(args.csv_path, index=False)
        print(f"CSV file saved to {args.csv_path}")

    split_training_testing(args.destination_folder, args.split, args.use_class)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-i", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--image_keywords", "-ik", type=str, default=None, nargs="+", help="Keywords to search for image files.")
    parser.add_argument("--mask_keywords", "-mk", type=str, default=None, nargs="+", help="Keywords to search for mask files.")
    parser.add_argument("--destination_folder", "-o", type=str, default="data/structured", help="Folder to save the structured dataset.")
    parser.add_argument("--startIdx", "-si", type=int, default=0, help="Start index for patient IDs.")
    parser.add_argument("--maskName", type=str, default="mask.nii.gz", help="Name of the mask files.")
    parser.add_argument("--imageName", type=str, default="image.nii.gz", help="Name of the image files.")
    parser.add_argument("--save_csv", action="store_true", help="Save the paths to a CSV file.")
    parser.add_argument("--csv_path", type=str, default="dataset.csv", help="Path to save the CSV file.")
    parser.add_argument("--split", "-s", type=float, default=0.8, help="Percentage of data to use for training.")
    parser.add_argument("--use_class", default=-1, type=int, help="Use a specific class for the segmentation.")
    
    args = parser.parse_args()
    run(args)