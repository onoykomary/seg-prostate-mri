# MRI Prostate Segmentation project

This repository contains a deep learning pipeline for 3D MRI Prostate Segmentation using the PROMISE12 dataset [1]. 
The project is built with **PyTorch Lightning** for training, **MONAI** for medical image processing and network architecture (3D UNet), and **SimpleITK** for NIfTI/MHD preprocessing.

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/fc2035c8-2d83-4532-9d5f-0c1450c679ca">
  <img width="295" src="https://github.com/user-attachments/assets/72d2f7d7-4333-453a-837c-8350db121b10"><br>
</p>
<p align="center">
  <em>
    <b>Left:</b> a) Ground Truth, b) Model prediction.<br>
    <b>Right:</b> 3D reconstruction of segmented prostate gland (GT = green, pred = red).
  </em>
</p>

## Features
* **Preprocessing:** N4 Bias Field Correction, Spacing Resampling (1x1x1 mm).
* **Augmentations** via MONAI.
* **Patch-based trainig strategy** implemented with PyTorch Lightning and MONAI, **sliding window inference**.
* **Logging:** batch visualization callback and MLFlow integration.
* **Dockerization**.


## Performance Results

| Metric | mean ± std 
|--------|------------
| **Dice Score** | **0.88 ± 0.03** 
| **Hausdorff Distance 95% (mm)** | **4.35 ± 2.98**
| **Average Surface Distance (mm)** | **1.42 ± 0.41**
| **Volume Error (ml)** | **3.14 ± 2.61**

## Project Structure
```
.
├── configs
│   └── config.yaml             # Pipeline configuration
├── src
│   ├── callbacks
│   │   └── visualize_batch.py  # Callback to save training slices
│   ├── dataset
│   │   ├── augment.py          # MONAI transformation pipeline
│   │   ├── pl_dataset.py       # LightningDataModule for PROMISE12
│   │   └── preprocess.py       # SimpleITK preprocessing
│   └── train
│       ├── pl_wrap.py          # LightningModule 
│       └── run.py              # Main training script
├── data
│   ├── raw                     # Place raw PROMISE12 .mhd/.raw files here
│   └── processed               # Output of preprocess.py (.nii.gz)
|
├── Experiments                 # Logs
├── Dockerfile                 
├── docker-compose.yaml 
├── Makefile
├── pyproject.toml            
└── README.md

```

## Data Preparation
Download the PROMISE12 dataset. Place the raw files inside data/raw/train and data/raw/val. 

## Running with Docker
1. Build the image
```
docker build -t prostate-segmentation:latest .
```
2. Run preprocessing \
Note! Preprocessing takes approximately 80 minutes.
```
make preprocess
```
3. Run training 
```
make train
```
4. Stop containers
```
docker compose down
```

## References
1. Litjens G. et al. PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012 [Dataset]. Zenodo. DOI: 10.5281/zenodo.8026660.
