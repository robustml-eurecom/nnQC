import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from torch.amp import autocast
import torchvision
import random
from scipy.ndimage import binary_erosion
from PIL import Image
from medpy.metric import binary
from scipy import stats
from tqdm import tqdm
from .dataset import AddPadding, CenterCrop, OneHot, LDMDataset
from monai.inferers import SliceInferer
from torch.utils.data import DataLoader
import monai
from collections import defaultdict

from batchgenerators.augmentations.utils import resize_segmentation
from IPython.display import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def erode_ohe_mask(mask, target_pixels=8, iterations=20):
    mask_np = mask.cpu().numpy()
    eroded_mask_np = np.zeros_like(mask_np)
    iters = 0
    
    for i in range(mask_np.shape[0]):
        class_mask = mask_np[i, :, :]
        if target_pixels is not None:
            while class_mask.sum() > target_pixels:
                class_mask = binary_erosion(class_mask)
        else:
            while iters < iterations:
                class_mask = binary_erosion(class_mask)
                iters += 1
        eroded_mask_np[i, :, :] = class_mask
        
    eroded_mask = torch.from_numpy(eroded_mask_np).to(mask.device)
    return eroded_mask

def create_holes(mask, min_radius, max_radius, max_holes):
    mask_np = mask.cpu().numpy()
    modified_mask = mask_np.copy()
    non_zero_indices = np.transpose(np.nonzero(mask_np))

    # Randomly select a subset of the non-zero voxels
    selected_indices = random.sample(list(non_zero_indices), min(max_holes, len(non_zero_indices)))

    for idx in selected_indices:
        radius = random.randint(min_radius, max_radius)

        # Create a spherical mask
        sphere_mask = np.zeros_like(mask_np)
        y,x,z = np.ogrid[-idx[0]:mask_np.shape[0]-idx[0], -idx[1]:mask_np.shape[1]-idx[1], -idx[2]:mask_np.shape[2]-idx[2]]
        mask_sphere = x*x + y*y + z*z <= radius*radius
        sphere_mask[mask_sphere] = 1

        # Create a hole at the current voxel
        modified_mask = np.where(sphere_mask, np.min(mask_np), modified_mask)  # assuming the background value is the minimum value

    modified_mask = torch.from_numpy(modified_mask).to(mask.device)
    return modified_mask


def evaluate_metrics(keys, prediction, reference, forceToOne=False):
    results = {}
    if forceToOne:
        prediction = np.where(prediction!=0, 1, 0)
        reference = np.where(reference!=0, 1, 0)
        
    for c,key in enumerate(keys,start=1):
        ref = np.copy(reference)
        pred = np.copy(prediction)

        ref = ref if c==0 else np.where(ref!=c, 0, ref)
        pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

        '''
        try:
            results["DSC_" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["DSC_" + key] = 0
        try:
            results["HD_" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["HD_" + key] = np.nan
        '''
        results["DSC_" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        results["HD_" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
    return results


def postprocess_image(image, info, current_spacing):
    postprocessed = np.zeros(info["shape"])
    crop = info["crop"]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(np.round(original_spacing[1:] / current_spacing[1:] * original_shape[:2]).astype(int)[::-1])
    image = np.argmax(image, axis=1)
    image = np.array([torchvision.transforms.Compose([
            AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot(info['classes'])
        ])(slice) for slice in image]
    )
    image = resize_segmentation(image.transpose(1,3,2,0), image.shape[1:2]+original_shape,order=1)
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed
  
  
def testing(ae, test_loader, patient_info, folder_predictions, folder_out, current_spacing):
    ae.eval()
    with torch.no_grad():
        results = {}
        for patient in test_loader:
            id = patient.dataset.id
            prediction, reconstruction = [], []
            for batch in patient: 
                
                if test_loader.condition:
                    condition = batch[1].to(device).float()
                    batch = {"prediction": batch[0].to(device)}
                    batch["reconstruction"] = ae.forward(batch["prediction"], condition)   
                else:
                    batch = {"prediction": batch.to(device)}
                    batch["reconstruction"] = ae.forward(batch["prediction"])
                
                if ae.n_classes > batch['reconstruction'].shape[1]:
                        batch['prediction'] = batch['prediction'][:,4:8,:,]
                
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
            
            reconstruction = postprocess_image(reconstruction.cpu().numpy(), patient_info[id], current_spacing)
            results["patient{:04d}".format(id)] = evaluate_metrics(
                ae.keys[1:],
                nib.load(os.path.join(folder_predictions, "patient{:04d}.nii.gz".format(id, patient_info[id]))).get_fdata(),
                reconstruction,
                forceToOne=test_loader.forceToOne
            )
            nib.save(
                nib.Nifti1Image(reconstruction, patient_info[id]["affine"], patient_info[id]["header"]),
                os.path.join(folder_out, 'patient{:04d}.nii.gz'.format(id, patient_info[id]))
            )
    return results


def create_dict():
    return {
        'scans': [],
        'segmentations': [],
        'ldm_reconstructions': [],
        'gt_masks': [],
        'baseline_reconstructions': []
    }

def create_result_dict():
    return {
        'ldm': {},
        'gt': {},
        'baseline': {}
    }

class SubjectVolumeEvaluator:
    def __init__(self, keys):
        self.keys = keys
        self.subject_volumes = defaultdict(create_dict)
        self.subject_results = defaultdict(create_result_dict)

    def add_subject_data(self, subject_id, scan, segmentation, 
                          ldm_reconstruction, baseline_reconstruction, gt_mask):
        """
        Add data for a specific subject, accumulating 3D volumes
        """
        self.subject_volumes[subject_id]['scans'] = scan
        self.subject_volumes[subject_id]['segmentations'] = segmentation
        self.subject_volumes[subject_id]['ldm_reconstructions'] = ldm_reconstruction
        self.subject_volumes[subject_id]['baseline_reconstructions'] = baseline_reconstruction
        self.subject_volumes[subject_id]['gt_masks'] = gt_mask

    def compute_subject_metrics(self):
        """
        Compute metrics for each subject by aggregating their volumes
        """
        for subject_id, volumes in self.subject_volumes.items():

            # Compute metrics for the subject
            self.subject_results['ldm'][subject_id] = evaluate_metrics(
                self.keys[1:], 
                volumes['segmentations'].argmax(1), 
                volumes['ldm_reconstructions'].argmax(1)
            )
            self.subject_results['gt'][subject_id] = evaluate_metrics(
                self.keys[1:], 
                volumes['segmentations'].argmax(1), 
                volumes['gt_masks'].argmax(1)
            )
            self.subject_results['baseline'][subject_id] = evaluate_metrics(
                self.keys[1:], 
                volumes['segmentations'].argmax(1), 
                volumes['baseline_reconstructions'].argmax(1)
            )
        print("Finished computing subject metrics")
        print("Results:", self.subject_results)
        print()
        return self.subject_results


def ldm_testing(
    unet, ae, baseline_ae, feature_extractor, 
    inferer, scheduler, test_loaders, 
    start_idx, end_idx, keys, logs_folder
):
    unet.eval()
    feature_extractor.eval()
    baseline_ae.eval()
    ae.eval()

    subject_evaluator = SubjectVolumeEvaluator(keys)

    results_output = os.path.join(logs_folder, "measures")
    os.makedirs(results_output, exist_ok=True)
    
    temp_res = None   
    if os.path.exists("{}/total_scores.npy".format(results_output)):
        temp_res = np.load("{}/total_scores.npy".format(results_output), allow_pickle=True).item()
    
    with torch.no_grad():
        gt_loader, test_loader = test_loaders
        
        for batch_gt, batch_test in zip(gt_loader, test_loader):
            subject_id = batch_test['id'] + start_idx
            subject_id = subject_id.cpu().numpy()[0]
            
            if subject_id > end_idx:
                break
            
            print("Processing subject", subject_id)
            
            if temp_res is not None and subject_id in temp_res['ldm'].keys():
                print("Already processed")
                continue
            
            gt_masks = batch_gt['seg'][0].to(device)
            segmentations = batch_test['seg'][0].to(device)
            scans = batch_test['img'][0].to(device)
            
            # Process patient data
            segmentations, reconstruction, bad_reconstruction, gt_masks, scans = process_patient(
                scans, segmentations, gt_masks, 
                unet, ae, baseline_ae, feature_extractor,
                inferer, scheduler
            )
            
            # Add data to subject evaluator
            subject_evaluator.add_subject_data(
                subject_id, 
                scans, 
                segmentations, 
                reconstruction, 
                bad_reconstruction, 
                gt_masks
            )
            
            plot_patient_results(
                subject_id,
                scans,
                gt_masks.argmax(1),
                segmentations.argmax(1),
                bad_reconstruction.argmax(1),
                reconstruction.argmax(1),
                logs_folder
            )
            # Compute subject-level metrics
            subject_results = subject_evaluator.compute_subject_metrics()
            # append the dict to temp res if its not none else create a new one
            if temp_res is None:
                temp_res = subject_results
            else:
                for key in subject_results.keys():
                    temp_res[key].update(subject_results[key])
                    
            np.save("{}/total_scores.npy".format(results_output), temp_res)
    
    subject_results = subject_evaluator.compute_subject_metrics()
    np.save("{}/total_scores.npy".format(results_output), temp_res)
    
    return subject_results


def erode_random_slices(segmentations, prob):
    target_pixels = random.choice([8, 16, 32, 64])
    iterations = random.randint(13, 20)
    if prob < 0.6:
        num_slices = segmentations.shape[0]
        num_random_slices = random.randint(5, num_slices)  # Number of slices to erode
        random_slices = random.sample(range(num_slices), num_random_slices) # Randomly select slices
        for i in random_slices:
            if prob < 0.2:
                segmentations[i] = erode_ohe_mask(segmentations[i], None, iterations)
            elif prob < 0.5:
                segmentations[i] = create_holes(segmentations[i], 15, 20, 15)
            else:
                segmentations[i] = erode_ohe_mask(segmentations[i], target_pixels)  
    return segmentations


def process_patient(scans, segmentations, gt_masks, unet, ae, baseline_ae, feature_extractor, inferer, scheduler):
    prob = random.uniform(0, .7)
    scans, segmentations, gt_masks = scans.to(device), segmentations.to(device), gt_masks.to(device) 
    segmentations = erode_random_slices(segmentations, prob)
    print("Segmentations shape:", segmentations.shape)
    
    print("Extracting Opinion 1:")
    opinion1 = feature_extractor.feature_extractor.encode(scans)
    condition = opinion1.reshape(opinion1.shape[0], -1)[:, None, :]
    print("Opinion 1 shape:", condition.shape)
    print("------------------------------------")
    
    if baseline_ae is not None:
        print("Extracting Opinion 2:")
        opinion2 = baseline_ae.encoder(segmentations)
        bad_reconstruction = baseline_ae(segmentations)
        print("Bad reconstruction shape:", bad_reconstruction.shape)
        
        opinion2 = opinion2.reshape(opinion2.shape[0], -1)[:, None, :]
        print("Opinion 2 shape:", opinion2.shape)
        
        condition = torch.cat([condition, opinion2], dim=-1)
        print('Final condition shape:', condition.shape)
        print("------------------------------------")
    
    z = torch.rand(segmentations.shape[0], 3, 64, 64).to(device)
    condition = condition.squeeze(0)
    scheduler.set_timesteps(num_inference_steps=1000)
    print("Noise shape:", z.shape)
    
    with autocast(device_type='cuda', enabled=True):
        print("Generating masks:")
        generated_masks = inferer(
            z,
            diffusion_model=unet,
            scheduler=scheduler,
            autoencoder_model=ae,
            conditioning=condition
        )
    
    print()
    return (
        segmentations.cpu().numpy(), 
        generated_masks.cpu().numpy(), 
        bad_reconstruction.cpu().numpy(), 
        gt_masks.cpu().numpy(), 
        scans.cpu().numpy()
    )


def round_results(results, results_gt, results_bad, id):
    for key in results["patient{:04d}".format(id)].keys():
        results["patient{:04d}".format(id)][key] = round(results["patient{:04d}".format(id)][key], 3)
        results_gt["patient{:04d}".format(id)][key] = round(results_gt["patient{:04d}".format(id)][key], 3)
        results_bad["patient{:04d}".format(id)][key] = round(results_bad["patient{:04d}".format(id)][key], 3)


def print_patient_results(id, results, results_gt, results_bad):
    print("Patient {:04d}".format(id))
    print('GT res:', results_gt["patient{:04d}".format(id)])
    print('LDM res:', results["patient{:04d}".format(id)])
    print('Baseline res:', results_bad["patient{:04d}".format(id)])


def plot_patient_results(id, img, gt, prediction, bad_reconstruction, reconstruction, logs_folder):
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    slice_num = random.randint(0, img.shape[0]-1)
    ax[0].imshow(np.rot90(img[slice_num,0,:,:], 2), cmap="gray")
    ax[0].set_title("ROI cropped image")
    ax[0].axis("off")
    ax[1].imshow(np.rot90(gt[slice_num,:,:], 2), cmap="gray")
    ax[1].set_title("Ground truth")
    ax[1].axis("off")
    ax[2].imshow(np.rot90(prediction[slice_num,:,:], 2), cmap="gray")
    ax[2].set_title("Input Segmentation")
    ax[2].axis("off")
    ax[3].imshow(np.rot90(bad_reconstruction[slice_num,:,:], 2), cmap="gray")
    ax[3].set_title("AE Reconstruction")
    ax[3].axis("off")
    ax[4].imshow(np.rot90(reconstruction[slice_num,:,:], 2), cmap="gray")
    ax[4].set_title("LDM Generation")
    ax[4].axis("off")
    plt.savefig(os.path.join(logs_folder, "patient{:04d}_slice{}.png".format(id, slice_num)))


def plot_condition_grids(condition1, condition2):
    """
    Plot two 10x10 grids of 4x4 images side by side
    
    Parameters:
    condition1: numpy array of shape (100, 4, 4)
    condition2: numpy array of shape (100, 4, 4)
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Function to create the full grid for each condition
    def create_grid(condition):
        # Initialize the 40x40 grid (10x10 grid of 4x4 images)
        full_grid = np.zeros((40, 40))
        
        # Fill the grid with 4x4 images
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                # Place each 4x4 image in its position
                full_grid[i*4:(i+1)*4, j*4:(j+1)*4] = condition[idx]
        
        return full_grid
    
    # Create and plot grids
    grid1 = create_grid(condition1)
    grid2 = create_grid(condition2)
    
    # Plot condition 1
    im1 = ax1.imshow(grid1, cmap='gray')
    ax1.set_title('Condition 1')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Add gridlines every 4 pixels
    for x in range(0, 41, 4):
        ax1.axvline(x-0.5, color='red', linewidth=0.5, alpha=0.3)
        ax1.axhline(x-0.5, color='red', linewidth=0.5, alpha=0.3)
    
    # Plot condition 2
    im2 = ax2.imshow(grid2, cmap='gray')
    ax2.set_title('Condition 2')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Add gridlines every 4 pixels
    for x in range(0, 41, 4):
        ax2.axvline(x-0.5, color='red', linewidth=0.5, alpha=0.3)
        ax2.axhline(x-0.5, color='red', linewidth=0.5, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig


def display_image(img):
    img = np.rint(img)
    img = np.rint(img / 3 * 255)
    display(Image.fromarray(img.astype(np.uint8)))
  
  
def display_difference(prediction, reference):
    difference = np.zeros(list(prediction.shape[:2]) + [3])
    difference[prediction != reference] = [240,52,52]
    display(Image.fromarray(difference.astype(np.uint8)))
  

class Count_nan():
    def __init__(self):
        self.actual_nan = 0
        self.spotted_CA = 0
        self.FP_CA = 0
        self.total = 0
      
    def __call__(self, df): 
        df_AE = df[[column for column in df.columns if "p" in column]]
        df_GT = df[[column for column in df.columns if "p" not in column]]
        check_AE = np.any(np.isnan(df_AE.values), axis=1)
        check_GT = np.any(np.isnan(df_GT.values), axis=1)

        self.actual_nan += np.sum(check_GT)
        self.spotted_CA += np.sum(np.logical_and(check_GT, check_AE))
        self.FP_CA += np.sum(np.logical_and(np.logical_not(check_GT), check_AE))
        self.total += np.sum(np.any(np.isnan(df.values), axis=1))
      
    def __str__(self):
        string = "Anomalies (DSC=0/HD=nan): {}\n".format(self.actual_nan)
        string += "Spotted by CA: {}\n".format(self.spotted_CA)
        string += "False Positive by CA: {}\n".format(self.FP_CA)
        string += "Total discarded from the next plots: {}".format(self.total)
        return string
   
def process_results(keys, cae, models, folder_GT, folder_pGT):
    count_nan = Count_nan()
    plots = {}
    for model in models:
        GT = np.load(os.path.join(folder_GT, "{}.npy".format(model)), allow_pickle=True).item()
        pGT = np.load(os.path.join(folder_pGT, "{}_{}.npy".format(model, cae)), allow_pickle=True).item()

        df = pd.DataFrame.from_dict(
            GT, 
            orient='index', 
            columns=["{}_{}".format(measure, label) for measure in ["DSC", "HD"] for label in keys])
        for measure in list(df.columns):
            df["p{}".format(measure)] = df.index.map({
                patient: pGT[patient][measure] for patient in pGT.keys()
            })

        df = df.replace(0, np.nan)
        count_nan(df)
        df = df.dropna()

        for measure in ["DSC", "HD"]:
            for label in keys:
                if("GT_{}_{}".format(measure,label) not in plots.keys()):
                    plots["GT_{}_{}".format(measure,label)] = []
                    plots["pGT_{}_{}".format(measure,label)] = []
                plots["GT_{}_{}".format(measure,label)] += list(df["{}_{}".format(measure,label)])
                plots["pGT_{}_{}".format(measure,label)] += list(df["p{}_{}".format(measure,label)])
    print(count_nan)
    return plots

