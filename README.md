# tFUSFormer: Physics-guided super-resolution Transformer for simulation of transcranial focused ultrasound propagation in brain stimulation

## Overview
This repository contains the implementation of neural networks designed to predict the simulation of transcranial focused ultrasound (tFUS) propagation in brain stimulation. tFUS brain stimulation is a non-invasive medical technique that uses focused sound waves to modulate neuronal activity in specific brain regions, offering potential therapeutic and research applications.

## Features
- Fast super-resolution convolutional neural network (FSRCNN), squeeze-and-excitation super-resolution residual network (SE-SRResNet), super-resolution generative adversarial network (SRGAN), and tFUSFormer architecture for accurate prediction of focal region.
- Custom loss functions that combine MSE, IoU loss, and distance function to enhance prediction accuracy.
- Data pre-processing and loading modules for efficient handling of the tFUS simulation data.
- Evaluation metrics to assess model performance, including IoU score and distance D between maximum pressure points.

## Requirements
- python 3.9
- torch-1.11.0+cu113
- cudatoolkit-11.3.1

## Installation
To set up the project environment:
- Clone the repository: `git clone https://github.com/iangilan/tFUSFormer.git`

## Dataset
The dataset used in this project consists of:
- Due to the size of the dataset, we only included the pre-trained model and test datasets.
- Training dataset is not included.

## Usage
1. Locate your test dataset in your local storage.
2. Edit config.py according to the user's need.
3. Edit the data loaders `tFUS_dataloader.py` to load and preprocess the data. 
4. Train the model using `python train_model.py`.
5. Evaluate the model's performance on test data using `python test_model.py`.

## Model Architecture
- The `FSRCNN_1ch` model is a 3D FSRCNN that consists of encoder and decoder blocks, designed for extracting features and predicting a focal region.
- The `SESRResNet_1ch` model is a 3D SESRResNet, designed for extracting features and predicting a focal region.
- The `SRGAN_1ch` model is a 3D SRGAN, designed for extracting features and predicting a focal region.
- The `tFUSFormer_1ch` model is a 3D tFUSFormer, designed for extracting features and predicting a focal region.
- The `tFUSFormer_1ch` model is a 3D tFUSFormer, designed for extracting features and predicting a focal region.
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

For the manifold discovery and analysis (MDA) analysis, please cite the following paper:
- [Md Tauhidul Islam, Zixia Zhou, Hongyi Ren, Masoud Badiei Khuzani, Daniel Kapp, James Zou, Lu Tian, Joseph C. Liao and Lei Xing. 2023, "Revealing Hidden Patterns in Deep Neural Network Feature Space Continuum via Manifold Learning", Nature Communications, 14(1), p.8506.]([https://www.nature.com/articles/s41416-018-0185-8](https://www.nature.com/articles/s41467-023-43958-w#citeas))
- BibTeX formatted citation\
@Article{Islam2023,\
author={Islam, Md Tauhidul and Zhou, Zixia and Ren, Hongyi and Khuzani, Masoud Badiei and Kapp, Daniel and Zou, James and Tian, Lu and Liao, Joseph C. and Xing, Lei},\
title={Revealing hidden patterns in deep neural network feature space continuum via manifold learning},
journal={Nature Communications},\
year={2023},\
month={Dec},\
day={21},\
volume={14},\
number={1},\
pages={8506},\
doi={10.1038/s41467-023-43958-w}
}

## Contact
For any queries, please reach out to [Minwoo Shin](mjmj0210@gmail.com).

