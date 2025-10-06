# 🧠 Workplace Safety Monitoring using Computer Vision

## 📌 Project Overview

This project leverages **Computer Vision** and **Deep Learning** to automatically detect **safe** and **unsafe** workplace conditions.  
It aims to enhance **occupational safety**, **reduce human monitoring costs**, and **minimize compliance risks** by using image classification models built on **Convolutional Neural Networks (CNNs)** and **VGG16** architectures.

---

## 🎯 Objectives

- Automate the detection of unsafe workplace environments.  
- Improve compliance with occupational safety standards (e.g., OSHA).  
- Reduce manual supervision costs and human error.  
- Build a scalable, proactive monitoring solution that ensures employee safety.

---

## 🧩 Business Problem

Traditional workplace safety monitoring is **manual, reactive, and error-prone**, leading to:
- Increased accident rates and insurance costs.  
- Legal penalties for non-compliance.  
- Operational inefficiencies and reputational damage.

### Solution

Use **AI-powered computer vision** to automatically classify images as:
- **Safe** (0)
- **Unsafe** (1)

This enables **real-time monitoring** using camera feeds integrated with AI-driven alerts.

---

## 📊 Dataset Overview

| Attribute | Description |
|------------|-------------|
| Total Images | 631 |
| Image Size | 200 × 200 pixels |
| Color Mode | RGB |
| Classes | Safe (320) / Unsafe (311) |
| Data Split | Train (70%), Validation (15%), Test (15%) |

All images were standardized and preprocessed with **normalization** and **augmentation** for better generalization.

---

## 🧹 Data Preprocessing

1. **Data Cleaning** – Validated consistency of labels and image counts.  
2. **Normalization** – Rescaled pixel values from 0–255 to 0–1.  
3. **Grayscale Testing** – Checked model performance on reduced dimensions.  
4. **Augmentation** – Used `ImageDataGenerator` with:
   - Rotation (±20°)
   - Width/Height shift (10%)
   - Shear (10%)
   - Zoom (10%)
   - Horizontal & Vertical flips  
5. **Dataset Splitting** – Stratified split to maintain class balance.

---

## 🧠 Models Developed

### 🧩 Model 1: Basic CNN (Baseline)
- 3 Convolutional Layers + MaxPooling
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Accuracy ≈ 0.998
- Great baseline; strong generalization.

---

### 🧩 Model 2: VGG16 (Transfer Learning)
- Used pre-trained **VGG16** on ImageNet (frozen convolutional layers)
- Fine-tuned classification head
- Accuracy / Precision / Recall / F1 = 1.0
- Faster convergence, high accuracy, minimal overfitting.

---

### 🧩 Model 3: VGG16 + Feed Forward Neural Network (FFNN)
- Added dense layers (256 → 128 neurons) with **Dropout(0.5)**
- Reduced overfitting, improved stability
- Perfect metrics (1.0) across validation sets
- More robust generalization.

---

### 🧩 Model 4: VGG16 + FFNN + Data Augmentation (Final Model)
- Combined transfer learning, dropout, and augmentation.
- Most **deployment-ready model**.
- Accuracy, Precision, Recall, and F1 Score = **1.0**
- Excellent performance on unseen data and real-world variability.

---

## 📈 Model Comparison

| Model | Architecture | Accuracy | Precision | Recall | F1 Score | Remarks |
|--------|--------------|-----------|------------|----------|-----------|----------|
| Model 1 | Simple CNN | 0.998 | 0.998 | 0.998 | 0.998 | Strong baseline |
| Model 2 | VGG16 Base | 1.0 | 1.0 | 1.0 | 1.0 | Transfer learning, fast convergence |
| Model 3 | VGG16 + FFNN | 1.0 | 1.0 | 1.0 | 1.0 | Dropout improves generalization |
| **Model 4** | **VGG16 + FFNN + Augmentation** | **1.0** | **1.0** | **1.0** | **1.0** | **Best deployment-ready model** |

---

## 🚀 Key Results

- **Model 4** achieved **perfect scores** across training, validation, and test sets.  
- Data augmentation improved robustness and reduced overfitting.  
- Very few misclassifications, with high true positives and true negatives.  
- Reliable detection of unsafe conditions in unseen workplace environments.

---

## 💼 Business Impact

### ✅ Benefits
- Enhanced **safety compliance** and reduced accidents.  
- Lower **insurance and legal costs**.  
- Improved **employee morale** and **corporate reputation**.  
- **Automated surveillance** reduces reliance on human monitoring.

### ⚠️ Risks of Not Adopting
- Continued accidents and injuries.  
- Higher legal exposure and financial loss.  
- Regulatory non-compliance and reputational harm.

---

## 🧭 Recommendations

1. **Adopt Model 4** for real-time safety monitoring.  
2. **Pilot deployment** in one facility before full rollout.  
3. **Integrate with existing CCTV feeds** and compliance dashboards.  
4. **Establish continuous learning pipeline** — retrain with new unsafe scenarios.  
5. **Expand enterprise-wide** after successful pilot.

---

## 🔮 Future Enhancements

- Increase dataset size with more diverse, real-world images.  
- Incorporate video stream analysis for live safety detection.  
- Test advanced architectures like **ResNet**, **EfficientNet**, or **Vision Transformers**.  
- Add **Explainable AI (XAI)** modules for transparency (e.g., Grad-CAM heatmaps).  
- Combine with IoT sensors for holistic safety monitoring.

---

## 🗂️ Project Structure

📦 workplace-safety-cv/
├── data/ # Dataset (.npy, .csv)
├── notebooks/ # Jupyter notebooks
├── models/ # Saved models (CNN, VGG16, etc.)
├── src/
│ ├── preprocess.py
│ ├── train_cnn.py
│ ├── train_vgg16.py
│ └── evaluate.py
├── reports/ # Visualizations, EDA, results
├── requirements.txt
└── README.md
