# 🩻 Pneumonia Detection from Chest X-Ray Images using VGG16

<p align="center">
  <b>Deep Learning Based Pneumonia Detection System</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Keras-VGG16-red?logo=keras">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit">
  <img src="https://img.shields.io/badge/Test%20Accuracy-84.94%25-success">
</p>

---

## 📌 Project Overview

**Pneumonia Detection from Chest X-Ray Images using VGG16** is a deep learning project developed to classify chest X-ray images into two categories:

- ✅ NORMAL
- ⚠️ PNEUMONIA

The project uses the **VGG16 Convolutional Neural Network with Transfer Learning** to identify useful visual patterns in chest X-ray images.

A **Streamlit web application** is developed to provide an easy-to-use interface. Users can upload a chest X-ray image and receive a prediction along with the model confidence.

> ⚠️ **Disclaimer:** This project is developed for educational and academic purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## 🎯 Project Objectives

The main objectives of this project are:

- To detect pneumonia from chest X-ray images.
- To use the VGG16 deep learning model.
- To apply transfer learning for image classification.
- To preprocess chest X-ray images.
- To train and evaluate the deep learning model.
- To save the trained model.
- To develop a Streamlit web application.
- To classify uploaded X-ray images as NORMAL or PNEUMONIA.

---

## 📊 Dataset

The project uses the **Pediatric Chest X-Ray Pneumonia Dataset**.

The dataset contains chest X-ray images belonging to two classes:

```text
NORMAL
PNEUMONIA
```

### Dataset Structure

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset was downloaded using the Kaggle API and used for training and testing the model in Google Colab.

---

## 🧠 Model Used

### VGG16

VGG16 is a Convolutional Neural Network architecture used for image classification and feature extraction.

In this project, VGG16 is used with Transfer Learning. The pretrained network helps extract useful image features, which are then used to classify chest X-ray images.

**Model Input:** `224 × 224 × 3`

### Model Workflow

```text
Chest X-Ray Image
        ↓
Resize to 224 × 224
        ↓
VGG16 Preprocessing
        ↓
VGG16 Model
        ↓
Feature Extraction
        ↓
Classification
        ↓
NORMAL / PNEUMONIA
```

---

## 🔄 Project Workflow

```text
Dataset Collection
        ↓
Image Preprocessing
        ↓
Image Resizing
        ↓
VGG16 Preprocessing
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
Streamlit Deployment
        ↓
X-Ray Prediction
```

---

## 🖼️ Image Preprocessing

Before an X-ray image is passed to the VGG16 model, several preprocessing steps are performed.

### 1. Image Upload

The Streamlit application accepts:

- `.jpg`
- `.jpeg`
- `.png`

### 2. RGB Conversion

The uploaded image is converted into RGB format.

```python
image = Image.open(uploaded_file).convert("RGB")
```

### 3. Image Resizing

The image is resized to:

**224 × 224 pixels**

This matches the required VGG16 input size.

### 4. NumPy Conversion

The image is converted into a NumPy array.

### 5. Batch Dimension

A batch dimension is added before prediction.

### 6. VGG16 Preprocessing

The image is processed using:

```python
tf.keras.applications.vgg16.preprocess_input()
```

---

## 🏋️ Model Training

The model was trained using Google Colab with GPU support.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | VGG16 |
| Framework | TensorFlow / Keras |
| Input Size | 224 × 224 × 3 |
| Number of Classes | 2 |
| Classification | Binary |
| Epochs | 10 |
| Training Platform | Google Colab |
| Output Classes | NORMAL / PNEUMONIA |

---

## 📈 Model Performance

The trained model achieved the following results:

| Metric | Result |
|---|---|
| Training Accuracy | 93.55% |
| Validation Accuracy | 95.69% |
| Test Accuracy | 84.94% |
| Test Loss | 0.3607 |

### Final Test Accuracy

**84.94%**

The test accuracy was calculated using unseen images from the test dataset.

---

## 📊 Training Results

The model performance improved during the training process.

- **Training Accuracy:** 93.55%
- **Validation Accuracy:** 95.69%
- **Test Accuracy:** 84.94%
- **Test Loss:** 0.3607

The difference between training/validation and test performance shows why evaluation on unseen test data is important.

---

## 🌐 Streamlit Web Application

The trained VGG16 model is integrated into a Streamlit web application.

The application provides a simple interface for uploading and classifying chest X-ray images.

### Application Features

**🩻 1. Upload Chest X-Ray**

Users can upload a chest X-ray image in:
- JPG
- JPEG
- PNG

**🤖 2. VGG16 Prediction**

The uploaded image is automatically processed and passed to the trained VGG16 model.

**✅ 3. NORMAL Prediction**

If the model predicts a normal X-ray, the application displays:

```text
✅ NORMAL
```

**⚠️ 4. PNEUMONIA Prediction**

If the model predicts pneumonia, the application displays:

```text
⚠️ PNEUMONIA DETECTED
```

**📊 5. Model Confidence**

The application displays the model confidence percentage.

Example:

```text
Model Confidence
91.25%
```

**📌 6. Project Information**

The application displays:

```text
Model: VGG16
Input Size: 224 × 224
Test Accuracy: 84.94%
```

### 🖥️ Streamlit Application Workflow

```text
User Uploads X-Ray
        ↓
Convert to RGB
        ↓
Resize to 224 × 224
        ↓
VGG16 Preprocessing
        ↓
Load Trained Model
        ↓
Generate Prediction
        ↓
Calculate Confidence
        ↓
Display Result
```

---

## 🔢 Prediction Logic

The model produces a prediction value between 0 and 1.

The application uses a threshold of 0.5.

```python
if prediction >= 0.5:
    result = "PNEUMONIA"
    confidence = prediction * 100
else:
    result = "NORMAL"
    confidence = (1 - prediction) * 100
```

### Prediction Interpretation

```text
Prediction >= 0.5
        ↓
PNEUMONIA

Prediction < 0.5
        ↓
NORMAL
```

---

## 📂 Project Structure

```text
Pneumonia_Detection_VGG16/
│
├── app.py
├── pneumonia_vgg16.keras
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `pneumonia_vgg16.keras` | Trained VGG16 model |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from GitHub |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| TensorFlow | Deep Learning |
| Keras | Neural Network Development |
| VGG16 | Image Classification |
| NumPy | Numerical Operations |
| Pillow | Image Processing |
| Streamlit | Web Application |
| Google Colab | Model Training |
| Kaggle | Dataset |
| GitHub | Version Control |

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Pneumonia_Detection_VGG16.git
```

Replace `YOUR_USERNAME` with your GitHub username.

Then enter the project directory:

```bash
cd Pneumonia_Detection_VGG16
```

### Step 2: Create a Virtual Environment

Python 3.13 is recommended.

```bash
py -3.13 -m venv venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```bash
.\venv\Scripts\activate
```

After activation, the terminal should show:

```text
(venv)
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The main dependencies are:

```text
streamlit
tensorflow
numpy
pillow
```

You can install them using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

Open this address in your web browser.

---

## 🧪 How to Use

1. Start the Streamlit application.
2. Open the application in your browser.
3. Click **Upload Chest X-Ray Image**.
4. Select a `.jpg`, `.jpeg`, or `.png` X-ray image.
5. The uploaded X-ray will be displayed.
6. The image is resized to 224 × 224.
7. VGG16 preprocessing is applied.
8. The trained model generates a prediction.
9. The application displays either **NORMAL** or **PNEUMONIA**.
10. The model confidence is also displayed.

---

## 📊 Example Output

### NORMAL

**Prediction Result:** ✅ NORMAL

**Model Confidence:** 94.21%

### PNEUMONIA

**Prediction Result:** ⚠️ PNEUMONIA DETECTED

**Model Confidence:** 89.52%

The actual confidence depends on the uploaded X-ray image.

---


---

## 🔐 GitHub Security

Do not upload sensitive files to GitHub.

The following files should be excluded:

```text
kaggle.json
.env
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.streamlit/secrets.toml
```

### Recommended `.gitignore`

```text
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
kaggle.json
.env
.streamlit/secrets.toml
```

The Kaggle API file should never be uploaded because it may contain API credentials.

The complete X-ray dataset should also not be uploaded to GitHub because it can make the repository unnecessarily large.

---

## ⚠️ Limitations

This project has several limitations:

- The model is trained on a specific chest X-ray dataset.
- Performance may vary on images from different datasets.
- The model can produce false positive and false negative predictions.
- A high confidence score does not guarantee a medically correct diagnosis.
- The model has not been clinically validated.
- The system should not replace a qualified medical professional.
- Further testing with larger and more diverse datasets would be required for real-world medical applications.

---

## 🔮 Future Enhancements

The project can be improved in the future by adding:

- 🔹 Fine-tuning of VGG16 layers.
- 🔹 ResNet50 comparison.
- 🔹 DenseNet121 comparison.
- 🔹 EfficientNet comparison.
- 🔹 Improved image augmentation.
- 🔹 Class imbalance handling.
- 🔹 Grad-CAM visualization.
- 🔹 Explainable AI.
- 🔹 Larger and more diverse datasets.
- 🔹 External validation.
- 🔹 Cloud deployment.
- 🔹 Improved Streamlit interface.
- 🔹 Prediction history.
- 🔹 Downloadable prediction reports.

---


---

## 🏁 Conclusion

The **Pneumonia Detection from Chest X-Ray Images using VGG16** project successfully demonstrates the use of deep learning and transfer learning for chest X-ray image classification.

The VGG16 model was trained to classify X-ray images into two categories: NORMAL and PNEUMONIA. The model achieved a 93.55% training accuracy, 95.69% validation accuracy, and 84.94% test accuracy, with a test loss of 0.3607.
