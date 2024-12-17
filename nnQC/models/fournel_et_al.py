import numpy as np
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple
import scipy.ndimage as ndimage
import tqdm

import torch
import torch.nn as nn
from monai.networks.nets import UNet, ResNet
from monai.metrics import compute_hausdorff_distance
from monai.losses import GeneralizedDiceLoss
from medpy.metric.binary import hd95, dc
from .trainers import Metrics


class MetricType(Enum):
    DSC = "dsc"
    HD95 = "hd95"


class ExperimentMode(Enum):
    STANDARD = "standard"
    FAILED = "failed"


class MISAQC(nn.Module):
    #a resnet than a softmax layer
    def __init__(self, num_classes: int = 2):
        super(MISAQC, self).__init__()
        self.resnet = ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=2,
            n_input_channels=num_classes + 1,
            num_classes=num_classes  # Single output for quality score
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    

class SegmentationQCTrainer:
    def __init__(
        self,
        metric_type: MetricType,
        mode: ExperimentMode,
        num_classes: int,
        class_names: List[str],
        class_weights: Dict[int, float] = None,
        num_unet_variants: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer with flexible class configuration
        
        Args:
            num_classes: Number of classes (including background if present)
            class_names: List of class names
            class_weights: Optional weights for each class during metric computation
        """
        self.metric_type = metric_type
        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_weights = class_weights or {i: 1.0 for i in range(num_classes)}
        self.num_unet_variants = num_unet_variants
        self.device = device

    def train_unet_variants(self, train_loader) -> List[nn.Module]:
        """Train multiple UNet variants with different architectures"""
        models = []
        
        for i in range(self.num_unet_variants):
            # Create UNet with varying parameters
            # Create features starting from 4, 8, 16, 32 and doubling each time
            features = [(2**(j + 2))*(i + 1) for j in range(4)] 
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=self.num_classes,  # Flexible number of classes
                channels=features,
                strides=[2] * (len(features) - 1),
                dropout=0.2 if i % 2 == 0 else 0.3
            ).to(self.device)

            # Train the model
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = GeneralizedDiceLoss(
                include_background=True, 
                softmax=True
                )

            print()
            print(f"Training UNet variant {i}")
            progress_bar = tqdm.tqdm(range(50))
            for epoch in progress_bar:
                model.train()
                for batch in train_loader:
                    images, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    progress_bar.set_postfix({'Epoch': epoch, 'Loss': loss.item()})
            print(f"Finished training UNet variant {i}")
            models.append(model)
            print("-------------------")

        return models

    def generate_eroded_batch(
        self, 
        batch: np.ndarray, 
        erosion_sizes: List[int] = [2, 4, 8]
    ) -> np.ndarray:
        """Generate eroded versions of a batch of segmentations"""
        eroded_segs = []
        
        for seg in batch:
            erosion_size = np.random.choice(erosion_sizes)
            eroded = np.zeros_like(seg)
            
            # Skip background class if present (class_idx = 0)
            start_class = 1 if 0 in seg else 0
            for class_idx in range(start_class, self.num_classes):
                class_mask = seg == class_idx
                eroded_mask = ndimage.binary_erosion(
                    class_mask,
                    structure=np.ones((erosion_size, erosion_size))
                )
                eroded[eroded_mask] = class_idx
                
            eroded_segs.append(eroded)
            
        return np.stack(eroded_segs)

    
    def compute_metric(
        self, 
        pred: np.ndarray,
        target: np.ndarray,
        ) -> np.ndarray:
        """Compute either DSC or HD95 based on metric_type"""
        computer = Metrics()
        keys = ['Class {}'.format(i) for i in range(0, self.num_classes)]
        scores = computer(pred, target, keys)
        
        if self.metric_type == MetricType.DSC:
            # create a vector of the dice scores
            return np.array([scores[key + '_dc'] for key in keys])
        else:  # HD95
            if pred.sum() == 0 or target.sum() == 0:
                return [0.0 for _ in keys]  # Or handle empty masks as needed
            return np.array([scores[key + '_hd'] for key in keys])
        

    def generate_predictions(self, models: List[nn.Module], data_loader) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all UNet variants
        
        Args:
            models: List of trained UNet models
            data_loader: DataLoader containing validation/test data
        
        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        
        for idx, model in enumerate(models):
            model.eval()
            model_preds = []
            gts = []
            with torch.no_grad():
                for batch in data_loader:
                    images = batch["img"].to(self.device)
                    outputs = model(images)
                    model_preds.append(np.concatenate(
                        (outputs.cpu().numpy(), images), axis=1))
                    gts.append(batch["seg"].cpu().numpy())
            # Concatenate all predictions for this model
            predictions[f"model_{idx}"] = np.concatenate(model_preds, axis=0)
            print(f"Finished generating predictions for model {idx}")
            print()
            gts = np.concatenate(gts, axis=0)
        print("Shape of predictions:", predictions["model_0"].shape)
        print("Shape of ground truth:", gts.shape)
        print("-------------------")
        return predictions, gts
    
    
    def train_qc_resnet(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray,
        val_ratio: float = 0.1,
        batch_size: int = 50,
        num_epochs: int = 30
    ) -> nn.Module:
        """Train ResNet to predict quality metric scores"""
        
        # Prepare data
        print("Preparing data for QC ResNet training")
        X, y = [], []
        for model_name, preds in predictions.items():
            cur_X = []
            cur_y = []
            
            for pred, gt in zip(preds, ground_truth):
                # Calculate metric for each class
                class_metrics = []
                pred_mask = pred[:-1] 
                metric = self.compute_metric(pred_mask, gt)
                class_metrics.append(metric)
                
                cur_X.append(pred)
                cur_y.append(class_metrics)
            
            X.extend(cur_X)
            y.extend(cur_y)

        X = np.array(X)
        y = np.array(y)

        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print()
        
        # Split data
        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        val_size = int(val_ratio * num_samples)
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]

        # Create QC ResNet
        qc_model = MISAQC(self.num_classes).to(self.device)

        optimizer = torch.optim.Adam(qc_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Training loop
        print("Training QC ResNet")
        progress_bar = tqdm.tqdm(range(num_epochs))
        for epoch in progress_bar:
            qc_model.train()
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                X_batch = torch.FloatTensor(X[batch_indices]).to(self.device)
                y_batch = torch.FloatTensor(y[batch_indices]).to(self.device)

                # Add eroded examples in failed mode
                if self.mode == ExperimentMode.FAILED:
                    eroded_X = self.generate_eroded_batch(X[batch_indices])
                    eroded_X = torch.FloatTensor(eroded_X).to(self.device)
                    X_batch = torch.cat([X_batch, eroded_X])
                    
                    # Calculate metrics for eroded examples
                    eroded_y = []
                    for eroded, gt in zip(eroded_X.cpu().numpy(), ground_truth[batch_indices]):
                        class_metrics = []
                        start_class = 1 if 0 in gt else 0
                        for class_idx in range(start_class, self.num_classes):
                            eroded_mask = eroded == class_idx
                            gt_mask = gt == class_idx
                            metric = self.compute_metric(eroded_mask, gt_mask)
                            metric *= self.class_weights.get(class_idx, 1.0)
                            class_metrics.append(metric)
                        eroded_y.append(class_metrics)
                    eroded_y = torch.FloatTensor(eroded_y).to(self.device)
                    y_batch = torch.cat([y_batch, eroded_y])

                optimizer.zero_grad()
                #X_batch = X_batch.unsqueeze(1)
                outputs = qc_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({'Epoch': epoch, 'Loss': loss.item()})
                
        print("Finished training QC ResNet")
        return qc_model


    def run_pipeline(self, train_loader):
        """Run complete pipeline"""
        unet_models = self.train_unet_variants(train_loader)
        predictions, gts = self.generate_predictions(unet_models, train_loader)
        qc_model = self.train_qc_resnet(predictions, gts)
        return qc_model, predictions
    

from typing import Dict, List, Union
import pandas as pd

class QCModelTester:
    def __init__(
        self, 
        model: torch.nn.Module,
        class_names: List[str],
        metric_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.class_names = class_names
        self.metric_type = metric_type
        self.device = device
            
    def predict_quality(
        self, 
        slices: Union[np.ndarray, torch.Tensor],
        return_format: str = 'dataframe'
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Predict quality metrics for a batch of 2D slices
        
        Args:
            slices: Batch of 2D segmentation slices [B, H, W] or [B, 1, H, W]
            return_format: 'dataframe' or 'dict'
            
        Returns:
            Quality predictions in requested format
        """
        self.model.eval()
                
        # Get predictions
        with torch.no_grad():
            predictions = self.model(slices)
            predictions = predictions.cpu().numpy()
            
        # Format results
        results = {}
        for i, class_name in enumerate(self.class_names):
            if class_name.lower() != 'background':
                results[class_name] = predictions[:, i]
                
        if return_format == 'dataframe':
            df = pd.DataFrame(results)
            df['slice_idx'] = range(len(predictions))
            df['metric'] = self.metric_type
            return df
        return results

