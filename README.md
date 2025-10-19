# ♻️ Garbage Classification Project

A deep learning project that classifies garbage into three categories: **plastic**, **organic**, and **metal** using a Convolutional Neural Network (CNN). This project includes both a Jupyter notebook for model training and a Streamlit web application for interactive classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Web App](#running-the-web-app)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [License](#license)

## 🎯 Overview

This project implements a lightweight CNN model for automatic garbage classification to help with waste management and recycling efforts. The system can identify and categorize waste materials into three main categories:

- **Plastic**: Plastic bottles, bags, containers, etc.
- **Organic**: Food waste, biodegradable materials, etc.
- **Metal**: Cans, metal containers, foil, etc.

## ✨ Features

- **CNN Model**: Custom-built Convolutional Neural Network for image classification
- **Interactive Web App**: User-friendly Streamlit interface for real-time classification
- **Image Preprocessing**: Automatic image resizing and normalization
- **Confidence Scores**: Displays prediction confidence for each classification
- **Jupyter Notebook**: Complete training pipeline with data splitting and evaluation

## 📁 Project Structure

```
garbage_classification_01/
├── app.py                                  # Streamlit web application
├── Garbage_Classification_Project.ipynb   # Jupyter notebook for training
├── garbage_classifier.h5                  # Trained model (26MB)
├── .gitignore                             # Git ignore configuration
└── README.md                              # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yasiruchamuditha/garbage_classification_01.git
   cd garbage_classification_01
   ```

2. **Install required packages**
   ```bash
   pip install streamlit tensorflow pillow numpy matplotlib scikit-learn
   ```

   Or create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install streamlit tensorflow pillow numpy matplotlib scikit-learn
   ```

## 💻 Usage

### Training the Model

If you want to train the model from scratch or retrain with your own dataset:

1. **Open the Jupyter notebook**
   ```bash
   jupyter notebook Garbage_Classification_Project.ipynb
   ```

2. **Prepare your dataset**
   - Organize your images in folders: `dataset_full/plastic/`, `dataset_full/organic/`, `dataset_full/metal/`
   - The notebook will automatically split the data into train/validation/test sets (70%/15%/15%)

3. **Run all cells**
   - The notebook will train the CNN model and save it as `garbage_classifier.h5`
   - Training parameters:
     - Image size: 128x128 pixels
     - Batch size: 32
     - Epochs: 12
     - Classes: 3 (plastic, organic, metal)

### Running the Web App

To use the pre-trained model with the Streamlit web interface:

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Use the application**
   - Open your web browser (usually at `http://localhost:8501`)
   - Upload an image of garbage (JPG, JPEG, or PNG)
   - View the classification result and confidence score

## 🧠 Model Details

The garbage classification model is built using TensorFlow/Keras with the following architecture:

- **Input**: 128x128 RGB images
- **Architecture**: Convolutional Neural Network (CNN)
- **Output**: 3 classes (plastic, organic, metal)
- **Training**: 12 epochs with data augmentation
- **Model Size**: ~26 MB

The model uses:
- Image normalization (pixel values scaled to 0-1)
- Data augmentation for better generalization
- Softmax activation for multi-class classification

## 📊 Dataset

The model is trained on a custom dataset containing images of:
- Plastic waste (bottles, bags, containers)
- Organic waste (food scraps, biodegradable materials)
- Metal waste (cans, containers, foil)

**Dataset Structure:**
```
dataset_full/
├── plastic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── organic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── metal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 📦 Dependencies

The project requires the following Python packages:

- **streamlit**: Web application framework
- **tensorflow**: Deep learning framework
- **pillow**: Image processing
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities (for training)

## 📄 License

This project is part of an undergraduate Computer Vision coursework assignment.

## 🙏 Acknowledgments

This project was developed as part of the Computer Vision module coursework (Part A).

## 📧 Contact

For questions or feedback, please contact the repository owner.

---

**Note**: The pre-trained model (`garbage_classifier.h5`) is included in the repository. You can use it directly with the Streamlit app without training from scratch.
