# 🍱 Food Recognition System (Soft Computing Project)

This project implements a deep learning–based food image classification system using **InceptionV3**. It was developed as part of a soft computing course to demonstrate the application of convolutional neural networks (CNNs) in real-world computer vision problems.

---

## 🎯 Objective

The goal of this system is to classify food images into **101 categories** with high accuracy. It leverages transfer learning, K-Fold cross-validation, and data augmentation to improve performance and robustness.

---

## 🗂️ Project Structure

food-recognition-system/ │ ├── dataset/ # Contains 'train', 'val', 'test' subfolders (not included in repo) │ ├── train/ │ ├── val/ │ └── test/ │ ├── train.ipynb # Main notebook for training the model ├── splitting_rule.ipynb # Script to handle K-Fold splitting and dataset organization ├── model.keras # Final trained model (excluded from GitHub due to size) ├── requirements.txt # All dependencies ├── README.md └── .gitignore


---

## 🧠 Model Summary

- **Base Model:** InceptionV3 (pretrained on ImageNet)
- **Input Size:** 299 x 299 x 3
- **Output Classes:** 101 food categories
- **Top Layers Added:**
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(101, activation='softmax')
- **Activation Functions:**
  - ReLU for intermediate layers
  - Softmax for the output layer

---

## ⚙️ Key Features

### ✅ Dataset Splitting
- Dataset is divided into:
  - **80,800 training images**
  - **10,100 validation images**
  - **10,100 test images**
- Images are equally distributed across all 101 classes.
- **5-Fold Cross-Validation** is applied using custom splitting logic (`splitting_rule.ipynb`).

### ✅ Data Augmentation
To enhance generalization and reduce overfitting, the training set is augmented using:

- Rotation
- Width/height shift
- Zoom
- Horizontal flip

Implemented using `ImageDataGenerator` from Keras.

### ✅ Overfitting Control
To prevent overfitting and improve learning:
- `Dropout(0.5)` used in dense layers
- `EarlyStopping` to stop training when validation loss stops improving
- `ModelCheckpoint` to save the best-performing model
- `ReduceLROnPlateau` to dynamically adjust learning rate on validation loss plateaus

---

## 🧪 Evaluation Metrics

- **Training/Validation Accuracy and Loss per Epoch**
- **Training time per fold**
- **Final Model Size**
- **Confusion Matrix** and **Classification Report** per fold

---

## 🧾 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
