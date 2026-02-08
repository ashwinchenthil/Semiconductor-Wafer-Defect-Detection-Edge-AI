# Edge-AI Semiconductor Wafer Defect Detection

## Overview
This project presents an Edge-AI based semiconductor wafer defect detection system developed for the IESA DeepTech Hackathon. A lightweight CNN model was trained to classify wafer defects and converted to ONNX for edge deployment.

## Dataset
The model was trained using semiconductor wafer datasets:

WM-811K Wafer Dataset:  
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Additional Wafer Dataset:  
https://www.kaggle.com/datasets/arbazkhan971/semiconductor-wafer-defect-dataset

(Dataset not uploaded due to large size)

## Model (ONNX)
Download trained ONNX model here:  
PASTE_YOUR_DRIVE_LINK_HERE

## Code Structure
Phase_1/
- train_cnn.py → model training
- simple_cnn.onnx → trained model
- README.md → phase details

## How to Run
1. Prepare dataset in train/validation/test folders  
2. Run training:
   python train_cnn.py
3. Convert to ONNX:
   python convert_to_onnx.py

## Model Performance
Accuracy: ~99–100%  
Precision/Recall/F1: ~1.0  
Model size: 63.7 MB  

## Hardware/Platform
Trained on local CPU system using TensorFlow/Keras.  
ONNX exported for edge deployment on embedded AI hardware.

## ONNX Link Drive
https://drive.google.com/file/d/16wxHgHEHVLW1rJPdUaGUjo74Q45hbrFz/view?usp=drive_link

## Dataset Links
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
https://www.kaggle.com/datasets/arbazkhan971/semiconductor-wafer-defect-dataset
