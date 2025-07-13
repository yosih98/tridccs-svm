# TriDCCS-SVM: A Hybrid Model for Satellite Image Classification
This repository contains the implementation of TriDCCS-SVM, an image classification model that leverages the complementary strengths of three Convolutional Neural Networks (CNNs) combined with Support Vector Machines (SVMs) for the classification of high-resolution satellite images.

The model was developed as part of PhD research at Universiti Sains Islam Malaysia (USIM), under the academic guidance of Prof. Dr. Rosalina Abdul Salam. 
---
## Overview
The goal of this work is to enhance classification accuracy for satellite images using a multi-step pipeline:
1. Data preprocessing: Cleaning datasets, resizing images, and handling invalid entries.
2. Feature Extraction: Extracting deep features using three pretrained CNN architectures.
3. Feature Fusion: Combining features from all three networks into a high-dimensional vector.
4. Classification: Training SVM on the fused features to perform the final classification.
5. valuation:  Achieving high accuracy and outperforming previous studies.

This approach was tested on two benchmark datasets:
- SIRI-WHU Dataset: Contains 2,400 images (12 land-cover classes, 200 images per class)
- UC Merced Land Use Dataset: Contains 2,100 images (21 land-use classes, 100 images per class)
- Both datasets are split into 80% training and 20% testing, maintaining class balance.

Compared to the core papers and earlier studies, the proposed model shows a significant enhancement in classification performance, achieving accuracies of 96.25% on SIRI-WHU and 97.86% on UC Merced.

---
##  Structure
implementation-code-tridccs-svm/
|
|-- datasets/              # Input datasets (SIRI-WHU, UC Merced)
|   |-- siri_whu/
|   |   |-- agriculture/
|   |   |-- commercial/
|   |   +-- ...
|   +-- uc_merced/
|       |-- airplane/
|       |-- buildings/
|       +-- ...
|
|-- notebooks/             # Jupyter notebook with implementation
|   +-- tridccs_svm_yousef.ipynb
|     | -- code-pdf /            
|       +-- tridccs-svm-yousef-alsafadi.pdf
|     | -- code-html /            
|       +-- tridccs-svm-yousef-alsafadi.html
|-- outputs/               
|   |-- models/            # Trained SVM models
|   |-- processed/         # Extracted and fused features
|   |-- results/           # Evaluation metrics and CSV reports
|| -- Implementation Progress Report/            
|   +-- Presentation.pptx 
|
|-- requirements.txt       # Python dependencies
+-- README.md              # Project documentation

---
## (Getting Started)

## Requirements
This work was developed and tested with:
- Python 3.12.7
- TensorFlow 2.19.0
- Keras
- scikit-learn
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- tqdm
- joblib

## System Requirements
This implementation can run on standard desktop or laptop machines. However, for efficient feature extraction and model training, the following specifications are recommended:
- Processor (CPU): Quad-core Intel i5 / AMD Ryzen 5 or higher
- RAM: Minimum 8 GB (Recommended: 16 GB or more)
- GPU (Optional): NVIDIA GPU with CUDA support (Recommended: 4 GB VRAM or higher, e.g., GTX 1650, RTX series)
- Disk Space: At least 5 GB free for datasets and outputs
- Operating System:Windows 10/11, Ubuntu 20.04+, or macOS 11+
> Note: GPU acceleration is optional but strongly recommended for faster CNN feature extraction. Without GPU, the process may take longer.

## Prepare Datasets 
Download and extract the SIRI-WHU and UC Merced datasets.

## Install Dependencies
The required Python packages are specified in the notebook. Install them using:
pip install tensorflow scikit-learn opencv-python pillow tqdm matplotlib seaborn joblib psutil platformdirs natsort xgboost

## Environment Setup
   Install libraries, print system specifications (CPU, RAM, GPU), and set up directories.

## Data Loading & Inspection
   Load datasets and preprocess images.
   Visualize class distributions and image size statistics.
   Check for corrupt or invalid images.

## Feature Extraction
   Extract features using three pretrained CNNs:
     * ResNet50
     * DenseNet169
     * EfficientNetB0
   Normalize and scale features.

## Feature Fusion
   Concatenate CNN features into a single representation.
   Visualize feature maps, correlations, and distributions.

## Classification
   Train an SVM classifier with RBF kernel on fused features.
   Evaluate model performance (Accuracy, Precision, Recall, F1-Score).
   Save trained models.

## Visualization & Reporting
   Generate confusion matrices, per-class metrics, and charts.
   Export results as CSV, Excel, and plots.
---

## Results
Dataset	       Accuracy	     F1-Score
SIRI-WHU	        96.25%	         96.25%
UC Merced	     97.86%	         97.86%

Per-class metrics and confusion matrices are available under outputs/results/ and outputs/visualizations/.

## Outputs
Trained SVM models: outputs/models/svm/
Feature files: processed/features/
Evaluation results: outputs/results/
Visualizations: outputs/visualizations/

## Running the Notebook
Launch Jupyter Notebook and open:
notebooks/tridccs-svm-yousef-alsafadi.ipynb
Run the cells sequentially to reproduce the results.
---

Author:
Yousef H.S Alsafadi
PhD Candidate, Universiti Sains Islam Malaysia (USIM)
Email: yosih1998@gmail.com



