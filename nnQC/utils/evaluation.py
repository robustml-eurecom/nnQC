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
from .dataset import AddPadding, CenterCrop, OneHot

from batchgenerators.augmentations.utils import resize_segmentation
from IPython.display import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def erode_ohe_mask(mask, target_pixels=8):
    mask_np = mask.cpu().numpy()
    eroded_mask_np = np.zeros_like(mask_np)

    for i in range(mask_np.shape[0]):
        class_mask = mask_np[i, :, :]

        while class_mask.sum() > target_pixels:
            class_mask = binary_erosion(class_mask)

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

        try:
            results["DSC_" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["DSC_" + key] = 0
        try:
            results["HD_" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["HD_" + key] = np.nan
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

def ldm_testing(
    unet, 
    ae,
    baseline_ae, 
    feature_extractor, 
    inferer, 
    scheduler, 
    test_loaders,
    keys,
    logs_folder
):   
    unet.eval()
    feature_extractor.eval()
    baseline_ae.eval()
    ae.eval()
    
    with torch.no_grad():
        results = {'ldm': {}, 'gt': {}, 'baseline': {}}
        for patient_img, patient_mask, patient_gt in zip(test_loaders[0], test_loaders[1], test_loaders[2]):
            id = patient_mask.dataset.id
            print("Processing patient {:04d}".format(id))
            prediction, reconstruction, bad_reconstruction, gt, img = process_patient(
                patient_img, patient_mask, patient_gt, unet, ae, baseline_ae, feature_extractor, 
                inferer, scheduler, id, logs_folder
            )          
            
            results['ldm']["patient{:04d}".format(id)] = evaluate_metrics(keys[1:], prediction, reconstruction)
            results['gt']["patient{:04d}".format(id)] = evaluate_metrics(keys[1:], prediction, gt)
            results['baseline']["patient{:04d}".format(id)] = evaluate_metrics(keys[1:], prediction, bad_reconstruction)

            round_results(results['ldm'], results['gt'], results['baseline'], id)
            
            print_patient_results(id, results['ldm'], results['gt'], results['baseline'])
            plot_patient_results(id, img, gt, prediction, bad_reconstruction, reconstruction, logs_folder)
                
    return results


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


def process_patient(patient_img, patient_mask, patient_gt, unet, ae, baseline_ae, feature_extractor, inferer, scheduler, id, logs_folder):
    prediction, reconstruction, bad_reconstruction, gt, img = [], [], [], [], []
    for batch_idx, (scans, segmentations, gt_masks) in enumerate(zip(patient_img, patient_mask, patient_gt)):
        prob = np.random.rand()
        scans, segmentations, gt_masks = scans.to(device), segmentations.to(device), gt_masks.to(device) 
        eroded = False
        
        if prob < 0.5:
            eroded = True
            target_pixels = random.choice([8, 16, 32, 64])
            segmentations = torch.stack([erode_ohe_mask(mask, target_pixels) for mask in segmentations]) if prob < 0.5 else segmentations
        condition = feature_extractor.encode(scans).to(device)
        condition = condition.view(condition.shape[0], -1).unsqueeze(1)
        
        if baseline_ae is not None:
            embeddings = baseline_ae.encode(segmentations).to(device)     
            bad_mask = baseline_ae.decode(embeddings)
            embeddings = embeddings.view(embeddings.shape[0], -1).unsqueeze(1)           
            condition = torch.cat([condition, embeddings], dim=-1)
        
        z = torch.randn(scans.shape[0], 3, 64, 64).to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        
        with autocast(device_type='cuda', enabled=True):
            generated_masks = inferer.sample(
                input_noise=z, diffusion_model=unet, 
                scheduler=scheduler, autoencoder_model=ae,
                conditioning=condition
            )
        
        # split the 1600 condition in 2 parts and plot the each result 40x40 grid
        condition1 = condition[:, :, :1600]
        condition2 = condition[:, :, 1600:]
        # normalize the condition to be in the range [0, 1]
        condition1 = (condition1 - condition1.min()) / (condition1.max() - condition1.min())
        condition2 = (condition2 - condition2.min()) / (condition2.max() - condition2.min())
        
        #plot two different subplots" one with 10x10 grid in which each cell is a 4x4 image, the other with 10x10 grid in which each cell is a 4x4 image
        condition1 = condition1.view(-1, 100, 4, 4)[0].cpu().numpy()
        condition2 = condition2.view(-1, 100, 4, 4)[0].cpu().numpy()
        
        # two figures, one for each condition made by 10x10 grid of 4x4 images
        fig = plot_condition_grids(condition1, condition2)
        plt.savefig(os.path.join(logs_folder, "patient{:04d}_conditions_erosion.png".format(id))) if eroded else plt.savefig(os.path.join(logs_folder, "patient{:04d}_conditions.png".format(id)))
        
        reconstruction = torch.cat([reconstruction, generated_masks], dim=0) if len(reconstruction) > 0 else generated_masks
        prediction = torch.cat([prediction, segmentations], dim=0) if len(prediction) > 0 else segmentations
        bad_reconstruction = torch.cat([bad_reconstruction, bad_mask], dim=0) if len(bad_reconstruction) > 0 else bad_mask
        gt = torch.cat([gt, gt_masks], dim=0) if len(gt) > 0 else gt_masks
        img = torch.cat([img, scans], dim=0) if len(img) > 0 else scans
        
    reconstruction = reconstruction.argmax(1).cpu().numpy()
    prediction = prediction.argmax(1).cpu().numpy()
    bad_reconstruction = bad_reconstruction.argmax(1).cpu().numpy()
    gt = gt.argmax(1).cpu().numpy()
    img = img.cpu().numpy()
    
    return prediction, reconstruction, bad_reconstruction, gt, img


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
    slice_num = random.randint(0, img.shape[1]-1)
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
    plt.savefig(os.path.join(logs_folder, "patient{:04d}_slice5.png".format(id)))


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

