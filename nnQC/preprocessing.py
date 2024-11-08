import os
import numpy as np
from utils.preprocess import generate_patient_info, apply_preprocessing
import argparse

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
    if args.test not in patient_info:
        patient_info = {**patient_info, **generate_patient_info(args.raw_folder, patient_ids, args.start_idx)}
        np.save(os.path.join(args.folder_out, "patient_info.npy"), patient_info)
    
    non_valid = [id for id in patient_info if len(patient_info[id]["crop"])>3]
    print(non_valid)
    median_crop, spacing_target = compute_crop_spacing(patient_info)
    fingerprints['crop'] = median_crop
    fingerprints['spacing'] = spacing_target
    np.save(os.path.join(args.folder_out, "fingerprints.npy"), fingerprints)

    get_folder = lambda folder, id: os.path.join(folder, 'patient{:04d}'.format(id))
    split_folder = args.raw_folder.split('/')[-1]
    mask_folder_out = os.path.join(args.folder_out, split_folder)
    image_folder_out = os.path.join(args.folder_out, "img_" + split_folder)
    
    print('Preprocessing files...')
    apply_preprocessing(
        patient_ids, args.start_idx, patient_info, fingerprints['spacing'], args.raw_folder, mask_folder_out, 
        get_folder, lambda patient_info, id: "mask.nii.gz",
        raw=False)
    apply_preprocessing(
        patient_ids, args.start_idx, patient_info, fingerprints['spacing'], args.raw_folder, image_folder_out, 
        get_folder, lambda patient_info, id: "image." + args.imgfrmt,
        raw=True)
    print('Preprocessing complete')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--raw_folder', default='data/training', type=str, help='Folder containing mask images')
    parser.add_argument('--folder_out', default='preprocessed', type=str, help='Folder to save fingerprints')
    parser.add_argument('--test', '-t', action='store_true', help='Use test data')
    parser.add_argument('--imgfrmt', '-if', default='nii.gz', type=str, help='Format of the images')
    parser.add_argument('--start_idx', '-s', default=0, type=int, help='Start index')
    parser.add_argument('--use_class', default=-1, type=int, help='Which class to use (e.g., for FLAIR21 if set to 1 it will use background+liver.)')
    args = parser.parse_args()
    
    run(args)
    