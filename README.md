# AI-Powered Crop Quality Analysis Using Vision Systems

## Overview

This project develops an AI-driven system to classify crop health (Healthy vs. Diseased) using computer vision and deep learning. Leveraging a large agricultural image dataset, the system employs a Convolutional Neural Network (CNN) and explores transfer learning with advanced pre-trained models to enhance feature detection. It includes real-time analysis capabilities through camera integration, displaying crop health status dynamically. The project aims to advance precision agriculture by automating crop quality assessment, improving scalability and accuracy over traditional methods.

## Features

- **Crop Health Classification**: Classifies crops as Healthy or Diseased using deep learning models.
- **Baseline CNN**: A custom CNN for binary classification, optimized for the dataset.
- **Transfer Learning**: Experiments with advanced models (e.g., EfficientNetB0, VGG16) to improve feature extraction.
- **Real-Time Analysis**: Integrates with a webcam to display crop health status in real-time.
- **Image Preprocessing**: Applies resizing, normalization, and data augmentation for robust model training.

## Technologies

- **Python**: Core programming language.
- **Deep Learning Frameworks**: TensorFlow for model development and training.
- **Computer Vision Libraries**: OpenCV for image processing and real-time analysis.
- **Data Processing Tools**: Pandas, Scikit-learn for dataset handling and splitting.
- **Dataset**: A large agricultural image dataset (e.g., PlantVillage with 54,306 images).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Amith-S28/Health-of-a-Plant.git
   cd Health-of-a-Plant
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Requirements include:

   - `tensorflow>=2.10.0`
   - `opencv-python>=4.5.5`
   - `pandas>=1.5.0`
   - `scikit-learn>=1.2.0`
   - `numpy>=1.23.0`

4. **Download the Dataset**:

   - Obtain the PlantVillage dataset (or similar agricultural dataset) and place it in a directory (e.g., `data/plantvillage/color/`).
   - Update the `dataset_dir` path in the script to point to this directory.

## Usage

1. **Prepare the Dataset**:

   - Ensure the dataset is organized in subdirectories by class (e.g., `Apple___healthy`, `Apple___Black_rot`).
   - The script automatically labels images as "Healthy" or "Diseased" based on folder names.

2. **Run the Script**:

   ```bash
   python crop_quality_analysis_plantvillage.py
   ```

   - If a pre-trained model exists, it will load and start real-time analysis.
   - Otherwise, it will train a new CNN model and save it for future use.

3. **Real-Time Analysis**:

   - Connect a webcam to your device.
   - The script will display a live feed with crop health status ("Healthy" or "Diseased") overlaid.
   - Press `q` to exit the webcam feed.

4. **Model Training**:

   - The script trains a CNN from scratch if no model is found.
   - To experiment with transfer learning models (e.g., EfficientNetB0), modify the script to include the desired model (see code comments).

## Project Structure

```
crop-quality-analysis/
├── crop_quality_analysis_plantvillage.py  # Main script for training and real-time analysis
├── data/                                 # Directory for dataset (e.g., PlantVillage)
├── models/                               # Directory for saved models and checkpoints
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Results

- **Baseline CNN**: Achieved high accuracy on the dataset, demonstrating effectiveness for binary classification.
- **Transfer Learning Models**: Explored advanced models, with varying performance due to fine-tuning and preprocessing challenges.
- **Real-Time Capability**: Successfully classifies crop health in real-time using webcam input.

## Limitations

- Limited to leaf-based analysis; may not generalize to fruits or whole plants.
- Advanced models require careful fine-tuning and proper input resolution for optimal performance.
- Real-time analysis may lag on low-end devices without optimization.

## Future Work

- Integrate multi-modal data (e.g., multispectral imaging) for enhanced analysis.
- Develop cross-crop models for broader applicability.
- Optimize models for edge devices (e.g., Raspberry Pi) using quantization or lightweight architectures.

