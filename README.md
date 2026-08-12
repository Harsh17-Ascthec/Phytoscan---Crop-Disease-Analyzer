# 🔬 Phytoscan: AI-Driven Crop Disease Diagnostic Portal

### 🌾 Tech-A-Thon 4.0 — Problem Statement 1: Agro-Tech & Rural Development

Phytoscan is a production-grade, deep-learning-powered agricultural decision-support web application. Designed to mitigate rural crop losses, it enables farmers to instantly diagnose and address plant disease infections in **Pepper, Potato, and Tomato crops** by uploading photographs of diseased leaves. 

The application has been completely redesigned with a modern, responsive web interface styled in **BigHaat's** corporate palette (Vibrant Red-Orange, Gold, and Clean White) and features localized English recommendations, multi-candidate model predictions, and an integrated SQLite transaction logging database.

---

## 🌟 Key Features

*   **⚡ Real-Time Web Diagnostics**: Replaces legacy, desktop-only GUI wrappers with a responsive, browser-based drag-and-drop web portal built on Streamlit.
*   **🎯 Multi-Candidate Predictions**: Rather than just displaying a single winner, the system outputs the **Top-3 model prediction candidates** with percentage-based confidence scores and visual progress bars to support agricultural research.
*   **💊 Decoupled Remedies Library (`remedies.json`)**: Architectural separation of crop remedies from Python execution scripts. Maps all 15 trained crop classes to clean, standard English display names and detailed, integrated biological and chemical pest management strategies.
*   **📜 Farmer Scan History Dashboard**: Automatically logs every transaction (Timestamp, Filename, Diagnosed Disease, and Confidence) into a localized **SQLite database (`history.db`)** to help growers track and map disease spread trends over time.
*   **🎨 BigHaat Branding Integration**: Designed with BigHaat's signature corporate branding assets—utilizing `#E12A36` (Vibrant Red) as the primary brand color, `#FFB300` (Warm Gold) as an accent indicator, and `#F4F6F8` for structural container backgrounds to deliver an executive-level aesthetic.

---

## 🛠️ System Architecture & ML Pipeline

Phytoscan implements a state-of-the-art computer vision pipeline to preprocess and analyze leaf samples:

```
[Uploaded Leaf Image] 
         │
         ▼
[Image Preprocessing] ──► Resize to (48, 48) ──► Convert to NumPy Array (RGB)
         │
         ▼
[Deep Learning Model] ──► Sequential CNN Inference (auto_chloro_model.h5)
         │
         ├──► Primary Diagnosis + Actionable English Remedies (remedies.json)
         ├──► Top-3 Alternative Classification Probabilities
         └──► Transaction Logged to SQLite Database (history.db)
```

### Deep Learning Model Specifications
*   **Model Type**: Sequential Convolutional Neural Network (CNN) built in Keras/TensorFlow.
*   **Feature Extraction Layers**: 4 Conv2D blocks integrated with `BatchNormalization`, `Activation` (ReLU), `MaxPooling2D` to extract spatial micro-features, and `Dropout` layers to eliminate overfitting.
*   **Classification Head**: Dense fully-connected layers terminating in a 15-way `Softmax` activation function matching the PlantVillage category distribution.
*   **Optimizer**: Adam compiled with modern parameter syntax (`learning_rate=0.005`) and evaluated using Categorical Cross-Entropy loss.

---

## 📦 Project Directory Structure

Ensure your workspace is structured as follows before running or deploying the server:

```text
phytoscan/
├── app.py                   # Streamlit web server and database controller
├── remedies.json            # English-only remedies database
├── requirements.txt         # Package dependency list
├── auto_chloro_model.h5     # Pre-trained Sequential CNN weights (PlantVillage)
└── .streamlit/
    └── config.toml          # Native Streamlit brand styling configuration
```

---

## 🚀 Installation & Local Development

Follow these steps to run Phytoscan on your local development machine:

### 1. Set Up your Virtual Environment
Open your terminal inside the project root folder and run:
```bash
# Create a localized virtual environment
python -m venv .venv

# Activate the environment
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt)
.venv\Scripts\activate.bat
# On macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies
Ensure your pip package manager is updated, then install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the Web Server
Launch the Streamlit web application from your terminal:
```bash
streamlit run app.py
```
*   **Local UI URL**: `http://localhost:8501`
*   **Local Network URL**: `http://192.168.X.X:8501` (Use this URL to test the app live on your mobile phone or tablet connected to the same Wi-Fi network!)

---

## 🌐 Cloud Deployment Guide

Phytoscan can be deployed to the cloud for free to allow the Tech-A-Thon judges to interact with it online.

### Option A: Hugging Face Spaces (Recommended for ML Models)
1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Click **Spaces ➔ New Space**.
3. Choose **Streamlit** as the SDK and choose the free **CPU basic** hardware tier.
4. Go to **Files and versions ➔ Add file ➔ Upload files** and commit the following files:
   * `app.py`, `remedies.json`, `requirements.txt`, and `auto_chloro_model.h5`.
5. Hugging Face will automatically install your libraries and launch the app under a permanent public URL.

### Option B: Streamlit Community Cloud
1. Push your code repository to **GitHub** (ensure `auto_chloro_model.h5` is pushed). *Note: If your model is larger than 100MB, use Git LFS.*
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
3. Click **New app**, select your repository, specify the main path as `app.py`, and click **Deploy**.

---

## 📜 Evaluated Classes & Dataset Source
The core neural network is trained on the curated **PlantVillage** dataset, containing 16,000+ high-quality images across 15 distinct health states:
1. Pepper (Bell) - Bacterial Spot
2. Pepper (Bell) - Healthy
3. Potato - Early Blight
4. Potato - Healthy
5. Potato - Late Blight
6. Tomato - Bacterial Spot
7. Tomato - Early Blight
8. Tomato - Healthy
9. Tomato - Late Blight
10. Tomato - Leaf Mold
11. Tomato - Septoria Leaf Spot
12. Tomato - Two-Spotted Spider Mite
13. Tomato - Target Spot
14. Tomato - Tomato Mosaic Virus
15. Tomato - Yellow Leaf Curl Virus

---

*Phytoscan represents a robust technical pipeline designed to bring automated diagnostic intelligence to rural farming communities.*
