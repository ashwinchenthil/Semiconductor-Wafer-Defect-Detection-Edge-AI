# Phase 1 – CNN Model Development for Wafer Defect Detection

## Objective
To develop and evaluate a deep learning model for automated semiconductor wafer defect classification as part of the IESA DeepTech Hackathon.

## Dataset
WM-811K semiconductor wafer dataset and additional wafer defect images.
Images were processed in grayscale and organized into Train, Validation, and Test sets.

## Methodology
A lightweight Convolutional Neural Network (CNN) was trained from scratch for multi-class defect classification. The model was optimized for accuracy and edge deployability.

## Results
- Accuracy: ~99–100%
- Precision (weighted): 1.00
- Recall (weighted): 1.00
- F1-score: 1.00
- Test samples evaluated: 8323
- Model size: 63.7 MB

## Output
The trained model was converted to ONNX format for hardware-independent edge deployment.

## Platform
- Python, TensorFlow/Keras
- Local CPU-based training (no GPU/cloud)
