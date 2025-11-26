# 📘 Image Classification with Neural Networks

**CIFAR-100 Training + Transfer Learning on STL-10**
Project by *Aditya Marada*, *Jarno Balk*, and *Camille Niessink*
(Computer Vision 1 – Lab Project, 2025) 

---

## 📌 Overview

This project explores the performance of multiple neural network architectures for image classification:

* **Fully-connected networks (TwoLayerNet, FourLayerNet)**
* **Convolutional networks (ConvNet, ImprovedConvNet)**
* **Transfer learning** using CIFAR-100 → STL-10

All models were implemented in **PyTorch** and trained from scratch, with later fine-tuning on a subset of STL-10.

The complete experimental methodology, architectures, results, and visualizations (t-SNE, per-class accuracies) are documented in the accompanying report.

---

## 📚 Datasets

### **CIFAR-100**

* 32×32×3 color images, 100 classes
* Used for training and validating all base models
* Example superclass/subclass images shown in report (pages 12–13) 

### **STL-10 (5-class subset)**

Classes used: **bird, deer, dog, horse, monkey**
Sample images shown on page 14 of the report 

We perform **transfer learning** by adapting a pretrained CIFAR-100 CNN to this smaller dataset.

---

## 🏗️ Model Architectures

### **1. TwoLayerNet (Baseline FCN)**

* 2 fully-connected layers
* ReLU activations
* 23.5% validation accuracy

### **2. ConvNet (Baseline LeNet-style CNN)**

* Modified LeNet
* AvgPool, custom tanh
* 25.27% validation accuracy

### **3. FourLayerNet (Deeper FCN)**

* 4 fully-connected layers
* Dropout (p=0.3)
* **Performed poorly → 16.2% accuracy**

Flattening removes spatial structure → FCNs scale poorly for images.

### **4. ImprovedConvNet (Main Model)**

Key improvements:

* Added convolutional depth
* ReLU activations
* Batch Normalization after every conv layer
* MaxPool instead of AvgPool
* Dropout (p=0.5) in classifier head
* SGD + momentum + cosine LR decay

**Result: 66.29% validation, 66.05% test accuracy**
(See learning curves in report pages 6–7) 

---

## 🔄 Transfer Learning (CIFAR-100 → STL-10)

Steps:

1. Load best ImprovedConvNet trained on CIFAR-100
2. **Freeze convolutional base**
3. Replace classifier head with a 5-class layer
4. Train only the head for 50 epochs

**Results:**

* 76.05% peak validation accuracy
* 73.65% test accuracy

Feature-space clusters visualized via t-SNE (see page 8) 

---

## 📊 Final Results Summary

| Model               | Dataset   | Peak Val Accuracy | Final Test Accuracy |
| ------------------- | --------- | ----------------- | ------------------- |
| TwoLayerNet         | CIFAR-100 | 23.51%            | 23.51%              |
| ConvNet             | CIFAR-100 | 25.27%            | 25.27%              |
| FourLayerNet        | CIFAR-100 | 16.20%            | 16.20%              |
| **ImprovedConvNet** | CIFAR-100 | **66.29%**        | **66.05%**          |
| **Fine-tuned CNN**  | STL-10    | **76.05%**        | **73.65%**          |

Source: report table on page 10–11 

---

## 🧪 Reproducibility

The repository includes:

* Model definitions (TwoLayerNet, FourLayerNet, ConvNet, ImprovedConvNet)
* Training scripts
* Hyperparameter configurations
* Transfer learning pipeline
* t-SNE embedding script
* Utilities for loading CIFAR-100 and STL-10

---

## 📄 Report

A **full PDF report** is available in this repository containing:

* Detailed architecture diagrams
* All learning curves
* All experimental results
* t-SNE plots
* STL-10 qualitative visualizations
* CIFAR-100 appendix images

