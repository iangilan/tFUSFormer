# tFUSFormer: Physics-guided super-resolution Transformer for simulation of transcranial focused ultrasound propagation in brain stimulation

## Overview
This repository contains the implementation of a neural network designed to predict the simulation of transcranial focused ultrasound propagation in brain stimulation. tFUS brain stimulation is a non-invasive medical technique that uses focused sound waves to modulate neuronal activity in specific brain regions, offering potential therapeutic and research applications.

## Features
- 3D Encoder-decoder-based convolutional neural network (EDCNN), U-Net, and Attention U-Net architecture for accurate prediction of ablated area and temperature distribution.
- Custom loss functions that combine MSE, weighted MSE, and weighted Dice loss to enhance prediction accuracy.
- Data pre-processing and loading modules for efficient handling of RFA simulation data.
- Evaluation metrics to assess model performance, including MSE, RMSE, dice coefficient, and Jaccard index.

## Requirements
- python 3.9
- torch-1.11.0+cu113
- cudatoolkit-11.3.1

## Installation
To set up the project environment:
- Clone the repository: `git clone https://github.com/iangilan/tFUSFormer.git`

## Dataset
The dataset used in this project consists of:
- Temperature distribution data post-RFA treatment (https://drive.google.com/file/d/1F7OFzfXZdc6jGWc_qIpxW5WrgBBzYh_o/view?usp=sharing)
- Ablated area data post-RFA treatment (https://drive.google.com/file/d/1CDLMCfDLaI5SfMdX6DV5EJflgjRwjj9D/view?usp=sharing)
- Electrode location and geometry data during RFA treatment (https://drive.google.com/file/d/18rzSAqrPdOKl7YipzP73VnS7oK9d_-Ua/view?usp=sharing).
- Segmented breast tumor data obtained from MR images (https://drive.google.com/file/d/1O85XRSbVJly1kMyxfzIvbwV-84xo0nxS/view?usp=sharing).
> Note: MR images of breast cancer patients from a publicly available dataset ([Saha et al., 2018](https://www.nature.com/articles/s41416-018-0185-8)) were utilized to model tumor geometry.

## Usage
1. Locate your RFA dataset in your local storage.
2. Edit config.py according to the user's need.
3. Edit the data loaders `data_loader_Temp.py` and `data_loader_Dmg.py` for temperature distribution and damaged area, respectively, to load and preprocess the data. 
4. Train the model using `python train_Temp.py` or `python train_Dmg.py`.
5. Evaluate the model's performance on test data using `python test_Temp.py` or `python test_Dmg.py`.

## Model Architecture
- The `RFACNN` model is a 3D EDCNN that consists of encoder and decoder blocks, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The `RFAUNet` model is a 3D U-Net, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The `RFAAttUNet` model is a 3D Attention U-Net, designed for extracting features and predicting both temperature distribution and damaged (ablated) areas.
- The architectures of all models are defined in `models.py`.

## Custom Loss Function
The model uses a combined loss function (`new_combined_loss` in `utils.py`) incorporating MSE, weighted MSE, and weighted Dice loss to cater to the specific challenges in RFA thermal effect prediction.

## Evaluation
The model is evaluated based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Dice Coefficient, providing a comprehensive assessment of its prediction accuracy.

## Citation
If you use this tool in your research, please cite the following paper:
- [M. Shin, M. Seo, S. Cho, J. Park, J. Kwon, D. Lee, K. Yoon. "PhysRFANet: Physics-Guided Neural Network for Real-Time Prediction of Thermal Effect During Radiofrequency Ablation Treatment." *Arxiv*](https://arxiv.org/abs/2312.13947)
- BibTeX formatted citation\
@misc{shin2023physrfanet,\
      title={{PhysRFANet}: {P}hysics-Guided Neural Network for Real-Time Prediction of Thermal Effect During Radiofrequency Ablation Treatment},\
      author={Minwoo Shin and Minjee Seo and Seonaeng Cho and Juil Park and Joon Ho Kwon and Deukhee Lee and Kyungho Yoon},\
      year={2023},\
      eprint={2312.13947},\
      archivePrefix={arXiv},\
      primaryClass={eess.IV},\
      doi={10.48550/arXiv.2312.13947}
}
- [A. Saha, M. R. Harowicz, L. J. Grimm, C. E. Kim, S. V. Ghate, R. Walsh, M. A. Mazurowski, A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features, Br. J. Cancer. 119 (2018) 508–516.](https://www.nature.com/articles/s41416-018-0185-8)
- BibTeX formatted citation\
﻿@Article{Saha2018,\
author={Saha, Ashirbani and Harowicz, Michael R. and Grimm, Lars J. and Kim, Connie E. and Ghate, Sujata V. and Walsh, Ruth and Mazurowski, Maciej A.},\
title={A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 {DCE-MRI} features},\
journal={British Journal of Cancer},
year={2018},\
month={Aug},\
day={01},\
volume={119},\
number={4},\
pages={508-516},\
doi={10.1038/s41416-018-0185-8}
}

## Contact
For any queries, please reach out to [Minwoo Shin](mjmj0210@gmail.com).

