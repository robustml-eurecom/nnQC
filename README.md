# nnQC: Neural Network Quality Control for Medical Image Segmentation

A comprehensive toolkit for training, evaluating, and improving medical image segmentation models using diffusion-based approaches with cross-attention mechanisms.

## 🎯 Overview

nnQC provides tools for:
- Training autoencoder models for medical image latent representations
- Training diffusion models with cross-attention for segmentation refinement
- Evaluating model performance with comprehensive metrics
- Generating synthetic segmentation masks for quality control

## 🏗️ Architecture

The system consists of three main components:

1. **Autoencoder**: Encodes medical images and segmentation masks into latent space
2. **Diffusion Model**: Generates refined segmentation masks using cross-attention with CLIP features
3. **Evaluation Pipeline**: Comprehensive metrics computation including DSC, HD95, and correlation analysis

## 📦 Installation

### From Source

```bash
git clone https://github.com/yourusername/nnQC.git
cd nnQC
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- MONAI >= 1.2.0
- CUDA-capable GPU (recommended)

See `requirements.txt` for complete dependency list.

## 🚀 Quick Start

### Using as a Package

```python
import nnqc

# Train autoencoder
nnqc.train_autoencoder()

# Train diffusion model  
nnqc.train_diffusion()

# Run evaluation
results = nnqc.evaluate_validation_set(args)
```

### Command Line Interface

```bash
# Train autoencoder
nnqc-train-ae -c config/config_train_32g.json -g 2

# Train diffusion model
nnqc-train-diffusion -c config/config_train_32g.json -g 2

# Run inference/evaluation
nnqc-inference -c config/config_train_32g.json

# Evaluate validation set
nnqc-evaluate -c config/config_train_32g.json
```

### Alternative CLI (using module syntax)

```bash
# Train autoencoder
python -m nnqc.training.train_autoencoder -c config/config_train_32g.json

# Train diffusion  
python -m nnqc.training.train_diffusion -c config/config_train_32g.json

# Run inference
python -m nnqc.inference.inference -c config/config_train_32g.json
```

## 📁 Project Structure 