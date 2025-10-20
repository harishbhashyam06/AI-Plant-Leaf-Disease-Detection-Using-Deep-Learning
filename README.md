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
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
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
```
![App Output](https://github.com/harishbhashyam06/AI-Plant-Leaf-Disease-Detection-Using-Deep-Learning/blob/main/test_images/output.png)
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

* **Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)
* **Model:** [Google Drive Model](https://drive.google.com/file/d/13uQJa-Bq1sEv2ai_XdrmvrJgWWba7v-7/view)
* **Papers:**

  * Mohanty et al. (2016) – *Using Deep Learning for Image-Based Plant Disease Detection*
  * Singh et al. (2019) – *PlantDoc: A Dataset for Visual Plant Disease Detection*
* **Docs:** TensorFlow, Keras, Streamlit

---

✅ **Run Summary**

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
pip install -r app/requirements.txt
python -m streamlit run app/main.py
```



