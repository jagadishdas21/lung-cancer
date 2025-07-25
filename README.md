# Classification prediction from lung cancer histopathology images using unsupervised learning

![Header](hne-slides/slide-10.png)

## 📌 Overview

This project applies **unsupervised learning** techniques such as **K-Means Clustering** and **Principal Component Analysis (PCA)** on deep features extracted from **lung cancer histopathology images** using a pretrained **ConvNeXt** model. The goal is to cluster and visualize patterns in cancer subtypes without using labeled data.

---

## 🧪 Key Objectives

- 📷 Process and normalize histopathology images  
- 🔍 Extract deep features using `ConvNeXt-Base` pretrained model  
- 📉 Reduce feature dimensions using PCA  
- 📊 Cluster the reduced features with K-Means  
- 🧮 Evaluate cluster-label alignment with precision, recall, F1, and AUC metrics

---

## 🛠️ Technologies Used

| Library         | Purpose                                  |
|------------------|--------------------------------------------|
| PyTorch          | Model loading, training, and inference     |
| torchvision      | Image preprocessing and dataset loading    |
| ConvNeXt         | Pretrained model for feature extraction    |
| scikit-learn     | PCA, K-Means, evaluation metrics           |
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
- PCA reduces 1000+ features to 2 or 3 components
- Helps visualize cluster formation and explained variance

### 4️⃣ K-Means Clustering
- Clusters are formed on PCA-reduced features
- The elbow method helps determine the optimal number of clusters

### 5️⃣ Cluster Evaluation
- Confusion matrix and classification metrics compare clusters with true labels
- Precision, Recall, F1 Score, and AUC are calculated for evaluation

## 📈 Results

| 🔍 Result Type           | 📝 Description                                  | 🖼️ Preview |
|--------------------------|------------------------------------------------|------------|
| 🧪 **PCA Variance Plot** | Displays how much variance is retained by each principal component | ![PCA Variance](images/pca_variance.png) |
| 📊 **K-Means Clustering**| 2D visualization of clusters formed on PCA-reduced feature space | ![K-Means Clustering](images/kmeans_clusters.png) |
| 📉 **Confusion Matrix**  | Comparison between true labels and predicted clusters | ![Confusion Matrix](images/confusion_matrix.png) |
| 📈 **Metric Scores Table** | Summary of Precision, Recall, F1 Score, AUC | ![Metrics Table](images/metrics_table.png) |

---



