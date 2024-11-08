# nnQC: a self-adapting method for quality control in organ segmentation.

## How-to:

### 1. Data preparation:
-  Install the required packages:
```bash
pip install -r requirements.txt
```
- Run the script on your dataset X. Be aware: the script is based on keywords, you may modify a little bit the X folder structure to fit the script. At least by creating a folder with all the needed sparse files.
```bash
python nnQC/data_preparation.py
```
_See the script for the arguments needed_

- The script will generate a new folder named `data_X` with the following structure:
```
nnQC
│
├── data
│   ├── data_X
│   │   ├── training
│   │   │   ├── patient0000
│   │   │   │   ├── image.nii.gz (or image.format-you-want)
│   │   │   │   └── mask.nii.gz
│   │   │   └── ...
│   │   ├── testing
│   │   │   ├── patient0000
│   │   │   │   ├── image.nii.gz (or image.format-you-want)
│   │   │   │   └── mask.nii.gz
│   │   │   └── ...
```
- Run the script to preprocess the data for the nnQC model:
```bash
python nnQC/preprocessing.py
```
_See the script for the arguments needed_

You are now ready to train the nnQC model:
### 2. First stage training:
```bash
python nnQC/train_AE.py
```
_See the script for the arguments needed_

The script is the same for each feature Expert and the VAE-GAN. You just need to change the `--model` argument.

### 3. Second stage training:
```bash
python nnQC/train_ldm.py
```
_See the script for the arguments needed_

The script will train the LDM model using the pretrained experts and the latent-space generator/decoder from the pretrained VAE-GAN.

### 4. Quality control/Evaluatio:
```bash
python nnQC/test.py
```
_See the script for the arguments needed_
