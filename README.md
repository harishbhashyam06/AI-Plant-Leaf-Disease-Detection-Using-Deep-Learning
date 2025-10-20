# 🌿 Plant Leaf Disease Detection Using Deep Learning

## 📘 1. Overview

This project detects **plant leaf diseases** from uploaded images using **deep learning models** built with TensorFlow and Keras.
A simple and interactive **Streamlit web app** allows users to upload leaf images and receive disease predictions along with confidence scores.

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

The dataset used for training and validation is the **PlantVillage Dataset** from Kaggle:
🔗 **[PlantVillage Dataset - Kaggle](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)**

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

* Split automatically: **80% train**, **10% validation**, **10% test**
* Image size: **224×224 px**
* Normalized: Pixel values scaled to [0, 1]
* Classes: Auto-inferred from folder names

---

## 🧠 3. Trained Model

The best-performing model (**DenseNet121**) was selected for deployment.
Download the trained model from Google Drive below:

🔗 **[Trained Model - Google Drive Link](https://drive.google.com/file/d/13uQJa-Bq1sEv2ai_XdrmvrJgWWba7v-7/view)**

Once downloaded, place it inside:

```
app/trained_model/plant_disease_prediction_model.h5
```

---

## ⚙️ 4. Environment Setup

### 🧩 Step 1 — Create and Activate Virtual Environment

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

> 🟢 Ensure your PowerShell prompt starts with `(venv)` before continuing.

---

### 🧩 Step 2 — Install Required Packages

The dependencies are listed inside `app/requirements.txt`.

```bash
pip install -r app/requirements.txt
```

#### ✅ Example `app/requirements.txt`

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

---

### 🧩 Step 3 — Verify Installations

```bash
python -m pip show tensorflow
python -m pip show streamlit
```

If both display version info, setup is successful ✅

---

## 🧱 5. Folder Structure

```
plant-disease-prediction-cnn-deep-leanring-project-main/
│
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5        # Pre-trained model
│   ├── class_indices.json                           # Label-class mapping
│   ├── config.toml                                  # Streamlit settings
│   ├── credentials.toml                             # (Optional for sharing)
│   ├── Dockerfile                                   # (Optional for deployment)
│   ├── main.py                                      # Streamlit web app
│   └── requirements.txt                             # Dependencies
│
├── model_training_notebook/                         # Jupyter notebook for training
│
├── test_images/                                     # Example test inputs
│   ├── test_apple_black_rot.JPG
│   ├── test_blueberry_healthy.jpg
│   └── test_potato_early_blight.jpg
│
├── venv/                                            # Virtual environment (local only)
└── README.md                                        # Project documentation
```

---

## 💻 6. Streamlit App Code (`app/main.py`)

```python
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# === Setup paths ===
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# === Load model and class labels ===
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# === Helper: Load & preprocess image ===
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# === Helper: Predict image class ===
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

# === Streamlit UI ===
st.title('🌿 Plant Leaf Disease Classifier')
st.write("Upload a leaf image to detect the disease using a trained deep learning model.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('Classify'):
            predicted_class, confidence = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
```

---

## 🚀 7. Run the Application

### Step 1 — Activate Virtual Environment

```bash
.\venv\Scripts\activate
```

### Step 2 — Run the App

```bash
python -m streamlit run app/main.py
```

You’ll see output similar to:

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

👉 Open your browser and go to [http://localhost:8501](http://localhost:8501)

---

## 🧪 8. Test the Model

Use sample images from `test_images/` to verify predictions.

```
test_images/
├── test_apple_black_rot.JPG
├── test_blueberry_healthy.jpg
└── test_potato_early_blight.jpg
```

### Example Output:

```
🌿 Plant Leaf Disease Classifier
--------------------------------
Prediction: Tomato Leaf Curl Virus
Confidence: 97.45%
```

---

## 🧩 9. Workflow Summary

1. **Load dataset** using TensorFlow’s `image_dataset_from_directory`.
2. **Train models** — Custom CNN, VGG16, MobileNetV2, DenseNet121.
3. **Evaluate models** — Track validation accuracy and loss.
4. **Save the best model** (DenseNet121).
5. **Deploy via Streamlit** for real-time inference.

---

## 🛠️ 10. Common Issues & Fixes

| Issue                                               | Cause                        | Fix                                     |
| --------------------------------------------------- | ---------------------------- | --------------------------------------- |
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow not installed     | `pip install tensorflow==2.15.0.post1`  |
| `streamlit not recognized`                          | Streamlit missing in venv    | `pip install streamlit`                 |
| `ValueError: numpy > 2.0`                           | Incompatible numpy           | `pip install numpy==1.26.4`             |
| `Model file not found`                              | Wrong or missing path        | Place model inside `app/trained_model/` |
| `Browser doesn’t open`                              | Streamlit didn’t auto-launch | Open `http://localhost:8501` manually   |

---

## 🧾 11. Quick Commands Summary

| Action               | Command                               |
| -------------------- | ------------------------------------- |
| Activate Virtual Env | `.\venv\Scripts\activate`             |
| Install Dependencies | `pip install -r app/requirements.txt` |
| Run Streamlit App    | `python -m streamlit run app/main.py` |

---

## 📚 12. References

* **Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)
* **Pretrained Model:** [Google Drive Model Link](https://drive.google.com/file/d/13uQJa-Bq1sEv2ai_XdrmvrJgWWba7v-7/view?usp=sharing)
* **Research Papers:**

  * Mohanty et al. (2016) – *Using Deep Learning for Image-Based Plant Disease Detection*
  * Singh et al. (2019) – *PlantDoc: A Dataset for Visual Plant Disease Detection*
* **Framework Docs:** TensorFlow, Keras, Streamlit

---

✅ **Run this project in 3 steps:**

```bash
.\venv\Scripts\activate
pip install -r app/requirements.txt
python -m streamlit run app/main.py
```


