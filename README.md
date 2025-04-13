# 🍱 Food Recognition System (Soft Computing Project)

This project implements a deep learning–based food image classification system using **InceptionV3**. It was developed as part of a soft computing course to demonstrate the application of convolutional neural networks (CNNs) in real-world computer vision tasks.

---

## 🎯 Objective

The goal of this system is to classify food images into **101 categories** using a transfer learning approach. The model is trained to recognize different types of food based on image data, utilizing modern deep learning techniques like fine-tuning, data augmentation, and regularization.

---

## 🗂️ Project Structure

<pre> ## 🗂️ Project Structure ``` food-recognition-system/ ├── dataset/ # Contains 'train', 'val', 'test' subfolders (not included in repo) │ ├── train/ │ ├── val/ │ └── test/ │ ├── train.ipynb # Main notebook for training the model ├── splitting_rule.ipynb # Notebook used to prepare dataset split ├── model.keras # Final trained model (excluded from GitHub due to size) ├── requirements.txt # List of Python dependencies ├── README.md └── .gitignore ``` </pre>


---

## 🧠 Model Architecture

- **Base Model:** InceptionV3 (pretrained on ImageNet)
- **Input Size:** 299 x 299 x 3
- **Output Classes:** 101 food categories
- **Top Layers Added:**
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(101, activation='softmax')
- **Activation Functions:**
  - ReLU for dense layers
  - Softmax for the output layer

---

## ⚙️ Features & Workflow

### 📁 Dataset Splitting

- Total images:
  - **80,800 training**
  - **10,100 validation**
  - **10,100 test**
- All images are evenly distributed across 101 classes.
- Splitting logic is handled in `splitting_rule.ipynb`.

### 🧪 Preprocessing & Augmentation

- Preprocessing done using `ImageDataGenerator`:
  - Rescale
  - Rotation
  - Zoom
  - Width & height shift
  - Horizontal flip
- Applied only to the training dataset to prevent overfitting.

### 🛡️ Overfitting Control

The following techniques are used to minimize overfitting:

- **Dropout(0.5)** in the dense layer
- **EarlyStopping**: Stops training when validation loss plateaus
- **ModelCheckpoint**: Saves the best model during training
- **ReduceLROnPlateau**: Reduces learning rate if validation loss stops improving

---

## 📊 Evaluation Metrics

- Training & Validation **Accuracy** and **Loss**
- **Training time per epoch**
- **Final model size**
- Test accuracy on unseen data

---

## 🧾 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
