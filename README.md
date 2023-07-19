# Covid19-infection-Lung-segmentation-and-classification

## Introduction
The COVID-19 Lung Segmentation Project aims to develop an automated system for segmenting lung regions from CT scan images of patients with COVID-19. Accurate lung segmentation is crucial for various diagnostic and treatment applications in the context of COVID-19. This project utilizes deep learning techniques, specifically convolutional neural networks (CNNs), to perform pixel-wise segmentation of lung regions from CT scans.

## Project Overview
The project focuses on developing a robust and accurate lung segmentation model using a dataset of COVID-19 CT scan images. The main steps involved in the project workflow are as follows:

![image](https://github.com/Nehaasah/Covid19-infection-Lung-segmentation-and-classification/assets/102512172/8d044bdf-410c-4fb2-873b-da8a6c8ef9e7)

1. Data Preprocessing: The CT scan images are preprocessed to handle noise, artifacts, and variability in image quality. This includes resizing, normalization, and augmentation techniques to enhance the data and improve model performance.

2. Model Development: A U-Net architecture is employed for lung segmentation. The U-Net model is trained on the labeled CT scan images using a suitable loss function and optimization algorithm. Various hyperparameters are tuned to achieve optimal performance.

3. Model Evaluation: The trained model is evaluated on a separate validation set using appropriate evaluation metrics such as Dice coefficient, sensitivity, specificity, and intersection over union (IoU). The model's performance is assessed to ensure accurate lung segmentation.

4. Result Visualization: The lung segmentation results are visualized to demonstrate the effectiveness of the model. The original CT scan images are overlaid with the segmented lung regions to provide visual insights and aid in the interpretation of the results.

## Steps:

1. Prepare your input CT scan image in DICOM or NIfTI format.
2. Run the pre-processing steps to normalize and prepare the image for segmentation.
3. Load the trained model weights.
4. Pass the pre-processed image through the model for lung segmentation.
5. Post-process the segmentation output for visualization or further analysis.

## Features
- Automated lung segmentation from COVID-19 CT scan images.
- Pre-processing techniques for noise reduction and image enhancement.
- U-Net architecture for accurate pixel-wise segmentation.
- Model evaluation using performance metrics.
- Visualization of segmented lung regions overlaid on original CT scan images.

## Dataset
The dataset used in this project consists of COVID-19 CT scan images obtained from [source]. The dataset includes labeled lung masks corresponding to each CT scan image, which serve as ground truth for training and evaluation.

## Model Architecture
The lung segmentation model employs the U-Net architecture, which consists of an encoder and a decoder. The encoder extracts high-level features from the input CT scan image, while the decoder performs upsampling and combines features from different levels to generate the segmentation map. The U-Net model is trained end-to-end using a suitable loss function, such as binary cross-entropy or Dice loss.

------------------------pic

## Results
The trained lung segmentation model achieves high accuracy and precision in segmenting lung regions from COVID-19 CT scan images. The model's performance is evaluated on a separate validation set, and the results demonstrate its effectiveness in accurately delineating lung boundaries.

Sample visualizations of the original CT scan images with overlaid lung segmentations are shown below:
------------------------pic
