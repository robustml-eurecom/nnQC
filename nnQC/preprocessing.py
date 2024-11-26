import os
import numpy as np
from utils.preprocess import generate_patient_info, apply_preprocessing
import argparse
from glob import glob
import nibabel as nib
import monai


def compute_crop_spacing(patient_info):
    crops = []
    for i in patient_info:
        crop = patient_info[i]["crop"]
        crop = [s.stop - s.start for s in crop]
        crops.append(crop)
    crops = np.array(crops)
    median_crop = np.max(crops, axis=0)
    median_crop = [int(np.ceil(c)) for c in median_crop]
    spacings = [patient_info[id]["spacing"] for id in patient_info]
    spacing_target = np.percentile(np.vstack(spacings), 50, 0)
    return median_crop, spacing_target


def get_patient_ids(folder):
    patients = os.listdir(folder)
    first_id = int(patients[0].split("patient")[1])
    last_id = int(patients[-1].split("patient")[1])
    return range(first_id, last_id+1)


def run(args):
    fingerprints = {}
    patient_ids = get_patient_ids(args.raw_folder)
        
    print(f"Found {len(patient_ids)} patients")
    
    os.makedirs(args.folder_out, exist_ok=True)
    if os.path.exists(os.path.join(args.folder_out, "patient_info.npy")):
        patient_info = np.load(os.path.join(args.folder_out, "patient_info.npy"), allow_pickle=True).item()
    elif not args.test:
        patient_info = generate_patient_info(args.raw_folder, patient_ids, args.start_idx)
        np.save(os.path.join(args.folder_out, "patient_info.npy"), patient_info)
    if args.test:
        patient_info = {**patient_info, **generate_patient_info(args.raw_folder, patient_ids, args.start_idx)}
        np.save(os.path.join(args.folder_out, "patient_info.npy"), patient_info)
 

    images = sorted(glob(os.path.join(args.raw_folder, "**", "*image.nii.gz")))
    segs = sorted(glob(os.path.join(args.raw_folder, "**", "*mask.nii.gz")))

    data = [{"img": img, "seg": seg, "id": i} for i, (img, seg) in enumerate(zip(images, segs))]
    print(data[0])

    print()
    print("Dataset loading")

    volume_ds = monai.data.CacheDataset(data=data)
    loader = monai.data.DataLoader(volume_ds, batch_size=1, num_workers=4)
    print()
    
    print('Preprocessing files...')
    fingerprints = apply_preprocessing(
        patient_ids,
        args.start_idx,
        patient_info,
        loader,
        args.raw_folder,
        args.folder_out,
        args.test
    )
        
    np.save(os.path.join(args.folder_out, "fingerprints.npy"), fingerprints)
    print('Preprocessing complete')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--raw_folder', default='data/training', type=str, help='Folder containing mask images')
    parser.add_argument('--folder_out', default='preprocessed', type=str, help='Folder to save fingerprints')
    parser.add_argument('--test', '-t', action='store_true', help='Use test data')
    parser.add_argument('--start_idx', '-s', default=0, type=int, help='Start index')
    args = parser.parse_args()
    
    run(args)
    