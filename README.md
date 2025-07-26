# Classification prediction from lung cancer histopathology images using deep learning

![Header](hne-slides/slide-10.png)

## 📌 Overview

This project applies **unsupervised learning** techniques such as **K-Means Clustering** and **Principal Component Analysis (PCA)** on deep features extracted from **lung cancer histopathology images** using a pretrained **ConvNeXt** model. The goal is to cluster and visualize patterns in cancer subtypes without using labeled data.

---

## 🧪 Key Objectives

- 📷 Process and normalize histopathology images  
- 🔍 Extract deep features using `ConvNeXt-Base` pretrained model  
- 📉 Reduce feature dimensions using PCA  
- 📊 Cluster the reduced features with K-Means  
- 🧮 Evaluate cluster-label alignment

---

## 🛠️ Technologies Used

| Library         | Purpose                                  |
|------------------|--------------------------------------------|
| PyTorch          | Model loading, training, and inference     |
| Torchvision      | Image preprocessing and dataset loading    |
| ConvNeXt         | Pretrained model for feature extraction    |
| Scikit-learn     | PCA, K-Means, evaluation metrics           |
| Matplotlib/Seaborn | Plotting confusion matrix and PCA graphs |

---

## 📥 Installation

### 1️⃣ Clone repository

```bash
git clone https://github.com/jagadishdas21/lung-cancer.git
cd lung-cancer
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
### 3️⃣ Run notebook

```bash
jupyter notebook convnext.ipynb
```
### 4️⃣ Data directory
```bash
./resized_data/
```
---

## 🔍 Methodology

### 1️⃣ Data Preprocessing
- Images are resized to `224x224`
- Dataset normalization using computed `mean` and `std`

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 2️⃣ Feature Extraction with ConvNeXt
- Pretrained ConvNeXt-Base model is used to extract high-dimensional embeddings from each image:
```python
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
```

### 3️⃣ PCA – Dimensionality Reduction
- PCA reduces 1000+ features to 2 components
- Helps visualize cluster formation and explained variance

### 4️⃣ K-Means Clustering
- Clusters are formed on PCA-reduced features
- The elbow method helps determine the optimal number of clusters

### 5️⃣ Cluster Evaluation

- Mean of Original Features
- Variance of Original Features

---

## 📈 Results

| 🔍 Result Type           | 📝 Description                                  | 🖼️ Preview |
|--------------------------|------------------------------------------------|------------|
| 1️⃣ **PCA** | Principle component analysis of features | ![PCA](results/pca-features.png) |
| 2️⃣ **Variance**| Explained variance ratio of first two components: 0.220, 0.096 | ![Variance](results/cumulative-explained-variance.png) |
| 3️⃣ **Confusion Matrix**  | Evaluating Clustering against Ground Truth | ![Confusion Matrix](results/confusion-matrix.png) |
| 4️⃣ **Confusion Matrix**  | Logistic regression on PCA features | ![Confusion Matrix](results/confusion-matrix-1.png) |
| 5️⃣ **Confusion Matrix**  | Random forest on 191 PCA features | ![Confusion Matrix](results/confusion-matrix-2.png) |
| 6️⃣ **Cluster Statistics** | Distribution of feature across clusters | ![Cluster Statistics](results/cluster-statistics.png) |

---
