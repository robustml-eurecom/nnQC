from models.fournel_et_al import (
    SegmentationQCTrainer, 
    QCModelTester,
    MetricType,
    ExperimentMode
)
from models.networks import get_device
from models.fournel_et_al import MISAQC
from utils.dataset import get_dataloader, get_test_dataloader
from utils.evaluation import erode_random_slices
import numpy as np
import argparse
import os
import torch
from torch.amp import autocast

import warnings 
warnings.filterwarnings("ignore")

def run(args):
    device, _ = get_device()
    
    output_dir = args.experiment_name
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = args.data_dir
    n_classes = args.n_classes
    metric = args.metric
    mode = args.mode
    
    #get metric and mode from MetricType and ExperimentMode
    metric = MetricType[metric]
    mode = ExperimentMode[mode]
    
    train_loader = get_dataloader(
        data_dir, 
        "train", 
        50, 
        n_classes, 
        sanity_check=True
    )
    
    ckpt = 'qc_resnet.pth'
    if os.path.exists(os.path.join(output_dir, ckpt)):
        qc_resnet = torch.load(os.path.join(output_dir, ckpt))
        print("Loaded model from checkpoint")
        
    else:
        resnet_trainer = SegmentationQCTrainer(
            metric_type=metric,
            mode=mode,
            num_classes=args.n_classes,  # Background + 1 structure
            class_names=[
                "Class {}".format(i) if i != 0 else "Background" for i in range(args.n_classes)
                ],
            num_unet_variants=args.num_unet_variants,
            device=device,
        )
        
        qc_resnet, _ = resnet_trainer.run_pipeline(
            train_loader
        )
    
        torch.save(qc_resnet, os.path.join(output_dir, 'qc_resnet.pth'))

    # Test the model
    test_loader = get_test_dataloader(
        data_dir, 
        None,
        n_classes, 
        sanity_check=True
    )
    
    # Initialize tester
    tester = QCModelTester(
        model=qc_resnet,
        class_names=["Class {}".format(i) if i != 0 else "Background" for i in range(args.n_classes)],
        metric_type=metric  
    )
    print()

    results_dict = {}
    # Test on a batch of slices
    for test_slices in test_loader:
        id = test_slices["id"].item()
        results_dict[id] = {}
        print("Testing on subject", id)
        
        prob = np.random.rand()
        test_slices = {k: v.to(device) for k, v in test_slices.items()}
        test_slices["img"] = test_slices["img"].squeeze(0)
        segmentations = test_slices["seg"].cpu().squeeze(0).numpy()
        segmentations = erode_random_slices(torch.tensor(segmentations), prob)
        segmentations = segmentations.to(device)
        input_slices = torch.cat(
            [test_slices["img"], segmentations], 
            dim=1
        )
        results = tester.predict_quality(input_slices, None)

        # Print results
        result_dir = args.result_dir
        os.makedirs(result_dir, exist_ok=True)
        results_dict[id] = results
        
    np.save(
        os.path.join(result_dir, f'results_{metric}.npy'), 
        results_dict, 
        allow_pickle=True
    )

    # Example output:
    #    LVM    LVC    LVT    LVPM  slice_idx metric
    # 0  0.92   0.95   0.85   0.88         0    DSC
    # 1  0.89   0.93   0.82   0.87         1    DSC
    # ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Resnet (Fournel et al.)')
    parser.add_argument('--data_dir', '-d', default='preprocessed', type=str, help='Path to data directory')
    parser.add_argument('--experiment_name', '-e', default=None, type=str, help='Name of experiment')
    parser.add_argument('--result_dir', '-r', default='results', type=str, help='Path to save results')
    parser.add_argument('--n_classes', '-n', default=2, type=int, help='Number of classes')
    parser.add_argument('--num_unet_variants', '-u', default=1, type=int, help='Number of U-Net variants')
    parser.add_argument('--metric',  default='DSC', type=str, help='Metric to use')
    parser.add_argument('--mode', default='STANDARD', type=str, help='Mode to use')
    
    args = parser.parse_args()
    
    run(args)


    
        