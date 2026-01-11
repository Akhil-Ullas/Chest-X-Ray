# Chest X-Ray Classification Using Deep Convolutional Neural Networks

This project implements an end-to-end **binary medical image classification pipeline** using **Convolutional Neural Networks (CNNs)** to distinguish between **NORMAL** and **PNEUMONIA** chest X-ray images.
The dataset is sourced from **Kaggle**, and the work emphasizes **architecture design, data augmentation, training stability, and generalization**, rather than leaderboard optimization.

---

## 📌 Problem Statement

Chest X-ray interpretation is a challenging visual task due to:

* High intra-class variability
* Class imbalance
* Subtle spatial patterns

The goal of this project is to explore how **deep CNNs** can learn hierarchical visual features from chest X-ray images while maintaining robustness and avoiding overfitting.

---

## 📂 Dataset

* **Source:** Kaggle (Chest X-Ray Images – Pneumonia)
* **Classes:** NORMAL, PNEUMONIA
* **Training distribution:**

  * NORMAL: 1,341 images
  * PNEUMONIA: 3,875 images
* **Class imbalance acknowledged and handled through regularization and augmentation**

---

## 🔧 Data Preparation & Pipeline

* Verified dataset structure and image integrity using `pathlib` and `PIL`
* Resized images to **200 × 200**
* Built efficient input pipelines using:

  * `tf.keras.image_dataset_from_directory`
  * Caching, shuffling, and prefetching (`tf.data`)
* Rebatched data to stabilize training dynamics

---

## 🔁 Data Augmentation (Medical-Context Aware)

Data augmentation was treated as **data-space regularization**, not a heuristic.

Applied transformations:

* Small rotations
* Translations
* Zoom
* Contrast variation

Avoided unsafe augmentations (e.g., horizontal flips) due to medical interpretation constraints.

---

## 🧠 Model Architecture

* Custom deep CNN (~2 million parameters)
* Progressive convolutional blocks (16 → 256 filters)
* Components used:

  * Conv2D + ReLU
  * Batch Normalization
  * Controlled Max Pooling
  * Global Average Pooling (instead of Flatten)
  * Fully connected layers with Dropout

**Why Global Average Pooling?**

* Reduces parameter count
* Improves generalization
* Limits overfitting on medical datasets

**Output:**

* Sigmoid activation
* Binary Cross-Entropy loss

---

## ⚙️ Training & Optimization

* Optimizer: Adam
* Early stopping based on:

  * Validation loss
  * Validation accuracy
* Trained for up to 40 epochs with controlled convergence
* Monitored training vs validation accuracy and loss curves

---

## 🔍 Evaluation & Inference

* Visualized training and validation metrics
* Implemented single-image inference pipeline
* Output includes:

  * Raw probability
  * Predicted class label

Used unseen test images to verify generalization behavior.

---

## 📚 Key Learnings

* CNNs effectively capture hierarchical features in chest X-ray images
* Data augmentation is critical for robustness on limited medical datasets
* Architectural depth, pooling strategy, and normalization significantly affect stability
* Medical imaging models require conservative preprocessing and validation discipline

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* PIL

---


