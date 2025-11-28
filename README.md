# ANN_PROJECT

# Image-Based Animal Type Classification (Cow vs Buffalo)

This project implements a deep learning–based system for **automatic classification of cattle and buffalo images** using **Convolutional Neural Networks (CNN)** and **Vision Transformers (ViT)**. The goal is to reduce human error in manual animal identification and support **AI-driven livestock management**.

## 📌 Project Overview

Animal Type Classification (ATC) is crucial for dairy management and government programs like **Rashtriya Gokul Mission (RGM)**. Traditional manual methods are slow and subjective. This project automates the process using modern computer vision techniques and compares CNN and ViT models for performance.

## 🚀 Features

* Binary classification: **Cow vs Buffalo**
* Comparison between **CNN** and **Vision Transformer (ViT)**
* Robust preprocessing and data augmentation
* High accuracy and strong generalization
* Streamlit-based web interface for real-time image prediction

## 🧠 Models Used

* **CNN (Baseline Model)**

  * 2 Convolution layers + MaxPooling
  * Dropout for regularization
  * Fast and lightweight

* **Vision Transformer (ViT-B/16)**

  * Patch-based image representation
  * Self-attention mechanism
  * Better handling of complex backgrounds and global features

## 📊 Dataset

Merged from Kaggle:

* **Indian Bovine Breeds Dataset** – 5042 images
* **Indian Buffalo Dataset** – 2784 images

**Total:** ~7800+ real-world images
Includes variations in breed, lighting, pose, background, and age.

## ⚙️ Methodology

1. Image preprocessing (224×224 resizing, normalization)
2. Data augmentation (rotation, flip, crop, scaling)
3. Train–validation–test split (70/20/10)
4. Model training with Adam / AdamW optimizers
5. Evaluation using accuracy, precision, recall, F1-score
6. Deployment using **Streamlit**

## 📈 Results

| Model              | Accuracy   |
| ------------------ | ---------- |
| CNN                | 88–90%     |
| Vision Transformer | **94–96%** |

✅ ViT outperformed CNN in accuracy, stability, and robustness.
✅ Better handling of shadows, background clutter, and similar-looking animals.

## 🧪 Tech Stack

* Python
* PyTorch
* Vision Transformer (ViT)
* CNN
* Streamlit
* Kaggle Datasets

## 🔮 Future Scope

* Multi-breed classification
* Mobile app integration
* Real-time CCTV/drone-based monitoring
* Advanced transformers (Swin, DeiT, ConvNeXt)
* Integration with platforms like **Bharat Pashudhan App**

## 👨‍💻 Team

* Pratham
* Saurabh Rai
* Tarun Bisht
* Yug Beniwal


