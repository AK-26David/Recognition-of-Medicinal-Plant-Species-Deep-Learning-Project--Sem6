# 🌿 Venation-Aware Medicinal Plant Species Classification

This repository presents a deep learning pipeline for classifying medicinal plant species using internal venation patterns and morphological features. The approach integrates RGB, venation, and edge information to enhance fine-grained leaf classification accuracy across 30 species.

---

## 📁 Repository Structure

├── final.ipynb # Complete end-to-end code (preprocessing, training, evaluation)

├── Model 1/ # Initial experiments with Modified ResNet-50

├── Model 2/ # Initial experiments with Custom VenationNet

├── Model 3/ # Initial experiments with Dual-Stream CNN

├── dataset/ # (Empty) Folder for placing input images

└── README.md # Project documentation


---

## 🧠 Implemented Models

### 1. Modified ResNet-50 (Transfer Learning)
- Based on ImageNet-pretrained ResNet-50
- Fine-tuned on RGB + venation + edge channels
- Serves as a baseline model

### 2. Custom VenationNet
- A lightweight custom CNN
- Designed for hierarchical venation pattern learning
- Utilizes multi-scale feature fusion

### 3. Dual-Stream CNN
- Processes RGB and venation/edge features in two parallel streams
- Fuses discriminative features using an attention mechanism
- Best overall performance on the dataset

---

## 🗂 Dataset

The dataset used is the **Medicinal Leaf Dataset**, publicly available on Mendeley Data.

🔗 [Medicinal Leaf Dataset on Mendeley Data](https://data.mendeley.com/datasets/nnytj2v3n5/1)

### Dataset Summary

| Property             | Value                          |
|----------------------|--------------------------------|
| Total Images         | 1,500                          |
| Number of Classes    | 30 Medicinal Plant Species     |
| Image Resolution     | 224 × 224                      |
| Channels Used        | RGB, Venation Map, Edge Map    |
| Labeling Method      | Folder-based Class Labeling    |
| Capture Context      | Natural daylight, garden-grown |
| Devices Used         | Samsung S9+, Canon Inkjet      |

📌 **Note**:  
The `dataset/` folder is left empty for size reasons. Please manually download and place the images from the [Mendeley Dataset](https://data.mendeley.com/datasets/nnytj2v3n5/1) into the appropriate folder structure before running the code.

---

## 📈 Performance Highlights

| Model              | Validation Accuracy | Macro F1 Score |
|-------------------|---------------------|----------------|
| Modified ResNet-50 | 73.35%              | 0.8444         |
| VenationNet        | 75.75%              | 0.7365         |
| **Dual-Stream CNN**| **78.00%**          | **0.7485**     |

- Evaluation includes per-class F1 scores, confusion matrices, and macro/weighted averages.
- The Dual-Stream CNN outperformed other models due to superior venation-aware representation learning.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- OpenCV
- scikit-learn
- matplotlib
- tqdm

You can install dependencies using:

pip install -r requirements.txt


✅ How to Run

Clone the repo:
git clone https://github.com/your-username/venation-plant-classification.git
cd venation-plant-classification
Download and place the dataset in the dataset/ folder following the folder-based structure.
Launch the notebook:
jupyter notebook final.ipynb
