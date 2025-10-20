# 🌿 Plant Leaf Disease Detection Using Deep Learning

This is a reviewed, fixed, and complete **README.md** for your Plant Leaf Disease Detection project using deep learning and Streamlit. It includes everything from setup to troubleshooting.

---

## 📘 1. Overview

This project detects **plant leaf diseases** from uploaded images using **deep learning models** built with **TensorFlow** and **Keras**. An interactive **Streamlit web app** allows users to upload leaf images and receive disease predictions with confidence scores.

### 🎯 Objectives

* Train multiple CNN-based architectures and identify the most accurate one.
* Deploy the trained model via a Streamlit web app.
* Assist farmers and researchers in early detection of crop diseases.

### 🧠 Models Used

| Model           | Type              | Description                                             |
| --------------- | ----------------- | ------------------------------------------------------- |
| **Custom CNN**  | From scratch      | Basic convolutional model for benchmarking              |
| **VGG16**       | Transfer learning | Pre-trained ImageNet model, fine-tuned for this dataset |
| **MobileNetV2** | Transfer learning | Lightweight, real-time prediction capable               |
| **DenseNet121** | Transfer learning | Deep and accurate model used for deployment             |

---

## 🌾 2. Dataset

### 📂 Dataset Source

Dataset used: **PlantVillage Dataset** from Kaggle.
🔗 [PlantVillage Dataset - Kaggle](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)

### 📁 Dataset Structure

```
Multi_Crop_Dataset/
└── dataset/
    ├── tomato/
    │   ├── Tomato_Healthy/
    │   ├── Tomato_Bacterial_Spot/
    │   └── ...
    ├── apple/
    ├── corn/
    └── ...
```

* **Split automatically**: 80% train, 10% validation, 10% test
* **Image size**: 224×224 px
* **Normalization**: Pixel values scaled to [0, 1]
* **Classes**: Auto-inferred from folder names

---

## 🧠 3. Trained Model

The best-performing model (**Custom CNN**) was selected for deployment.

📥 **Download Model**: [Google Drive Link](https://drive.google.com/file/d/13uQJa-Bq1sEv2ai_XdrmvrJgWWba7v-7/view)

After download, place it here:

```
app/trained_model/plant_disease_prediction_model.h5
```

---

## ⚙️ 4. Environment Setup

### Step 1 — Create and Activate Virtual Environment

**Windows (PowerShell):**

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

> Ensure `(venv)` appears at the start of your terminal line.

### Step 2 — Install Dependencies

```
pip install -r app/requirements.txt
```

#### Example `app/requirements.txt`

```
tensorflow==2.15.0.post1
keras==2.15.0
numpy==1.26.4
pillow>=10.0.0
opencv-python==4.10.0.84
streamlit==1.30.0
protobuf<5
h5py>=3.10
```

### Step 3 — Verify Installation

```bash
python -m pip show tensorflow
python -m pip show streamlit
```

If missing:

```bash
pip install tensorflow
pip install streamlit
```

---

## 🧱 5. Folder Structure

```
plant-leaf-disease-dl/
│
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5
│   ├── class_indices.json
│   ├── main.py
│   ├── config.toml
│   ├── credentials.toml
│   ├── Dockerfile
│   └── requirements.txt
│
├── model_training_notebook/
│   └── train.ipynb
│
├── test_images/
│   ├── test_apple_black_rot.JPG
│   ├── test_blueberry_healthy.jpg
│   └── test_potato_early_blight.jpg
│
├── venv/
└── README.md
```

---

## 💻 6. Streamlit Application (`app/main.py`)

```python
import os
import io
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

WORKDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKDIR, 'trained_model', 'plant_disease_prediction_model.h5')
CLASS_INDICES_PATH = os.path.join(WORKDIR, 'class_indices.json')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, 'r') as f:
        return json.load(f)

model = load_model()
class_indices = load_class_indices()

def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image_bytes):
    x = preprocess_image_bytes(image_bytes)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = class_indices.get(str(idx), f'class_{idx}')
    return label, prob, preds.flatten()

st.set_page_config(page_title='Plant Leaf Disease Classifier', page_icon='🌿')
st.title('🌿 Plant Leaf Disease Classifier')
st.write('Upload a leaf image to detect the disease using a trained deep learning model.')

uploaded = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    if st.button('Classify'):
        label, prob, raw = predict(uploaded.getvalue())
        st.success(f'**Prediction:** {label}\n\n**Confidence:** {prob*100:.2f}%')

        if st.checkbox('Show top-5 classes'):
            top5_idx = np.argsort(raw)[-5:][::-1]
            st.write({f'{class_indices.get(str(i), i)}': f'{raw[i]*100:.2f}%' for i in top5_idx})

        if prob < 0.6:
            st.info('Low confidence: try a clearer, single-leaf photo in good lighting.')
```

---

## 🚀 7. Run the Application

### Step 1 — Activate Virtual Environment

```
.\venv\Scripts\activate
```

### Step 2 — Run Streamlit App

```
python -m streamlit run app/main.py
```

Open in your browser: [http://localhost:8501](http://localhost:8501)

---

## 🧪 8. Test the Model

Use `test_images/` to verify predictions:

```
test_images/
├── test_apple_black_rot.JPG
├── test_blueberry_healthy.jpg
└── test_potato_early_blight.jpg
```

**Example Output:**

```
Prediction: Tomato Leaf Curl Virus
Confidence: 97.45%
```

---

## 🧩 9. Workflow Summary

1. Load dataset using `image_dataset_from_directory`
2. Train models — Custom CNN, VGG16, MobileNetV2, DenseNet121
3. Evaluate performance using accuracy and loss
4. Save best model (`.h5`)
5. Deploy via Streamlit for real-time use

---

## 🛠️ 10. Troubleshooting

| Issue                          | Cause                    | Fix                                            |
| ------------------------------ | ------------------------ | ---------------------------------------------- |
| `No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow==2.15.0.post1`         |
| `streamlit not recognized`     | Streamlit missing        | `pip install streamlit`                        |
| `ValueError: numpy > 2.0`      | Version conflict         | `pip install numpy==1.26.4`                    |
| `Model not found`              | Wrong path               | Place `.h5` in `app/trained_model/`            |
| `App reloads slowly`           | Model reloads each run   | Use `@st.cache_resource`                       |
| Wrong labels                   | JSON mismatch            | Recreate `class_indices.json` in correct order |

---

## 🧾 11. Quick Commands Summary

| Action               | Command                               |
| -------------------- | ------------------------------------- |
| Activate Virtual Env | `.\venv\Scripts\activate`             |
| Install Dependencies | `pip install -r app/requirements.txt` |
| Run Streamlit App    | `python -m streamlit run app/main.py` |

---

## 📚 12. References

**Research Papers**

* **Mohanty et al. (2016)** – Introduced deep CNNs (AlexNet, GoogLeNet) for plant disease detection using the PlantVillage dataset; achieved **~99% accuracy**, proving CNNs’ power in agriculture.
* **Singh et al. (2019)** – Released **PlantDoc**, a real-world dataset with diverse backgrounds, showing that lab-trained models need domain adaptation for real field conditions.

**Framework Docs**

* [**TensorFlow**](https://www.tensorflow.org/) – Core ML framework for training and deploying models.
* [**Keras**](https://keras.io/) – High-level API for fast deep learning model development.
* [**Streamlit**](https://docs.streamlit.io/) – Framework for deploying interactive ML web apps easily.

✅ *These references validate your dataset, model selection, and deployment approach scientifically and technically.*

---

## 🛳️ Deployment with Docker

Below is a production-ready **Dockerfile** setup for deploying your Streamlit + TensorFlow app.

```dockerfile
# 🌿 Base image with Python 3.10
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (for caching efficiency)
COPY app/requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy full application code
COPY app /app

# Expose Streamlit’s default port
EXPOSE 8501

# Create Streamlit configuration directory
RUN mkdir -p ~/.streamlit

# Copy optional Streamlit configuration files
COPY app/config.toml ~/.streamlit/config.toml
COPY app/credentials.toml ~/.streamlit/credentials.toml

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 🚀 Build and Run Instructions

```bash
# Build Docker image
docker build -t plant-disease-app .

# Run container exposing port 8501
docker run -p 8501:8501 plant-disease-app
```

Access the app locally at 👉 **[http://localhost:8501](http://localhost:8501)**
