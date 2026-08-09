# 🩻 Pneumonia Detection from Chest X-Ray Images using VGG16

<p align="center">
  <h2 align="center">🩻 AI-Based Pneumonia Detection System</h2>
  <p align="center">
    Pneumonia Detection from Chest X-Ray Images using VGG16 Transfer Learning
  </p>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG16-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b?logo=streamlit)
![Model](https://img.shields.io/badge/Model-VGG16-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-84.94%25-success)

</p>

---

## 📌 Project Overview

**Pneumonia Detection from Chest X-Ray Images using VGG16** is a deep learning project developed to classify chest X-ray images into two categories:

- ✅ **NORMAL**
- ⚠️ **PNEUMONIA**

The project uses the **VGG16 convolutional neural network with transfer learning**. A pretrained VGG16 model is used to extract useful image features, and the trained model is used to classify an uploaded chest X-ray image.

A **Streamlit web application** is also developed to provide an easy-to-use interface. Users can upload a chest X-ray image and receive the predicted class along with the model confidence.

> ⚠️ **Disclaimer:** This application is developed for educational and academic purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.

---

# 🎯 Project Objective

The main objective of this project is to develop a deep learning-based system that can classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

### Objectives

- To understand medical image classification using deep learning.
- To use VGG16 for image feature extraction.
- To apply transfer learning to a chest X-ray dataset.
- To preprocess X-ray images for VGG16.
- To train a binary image classification model.
- To evaluate the model using test data.
- To save the trained model.
- To develop a Streamlit web application.
- To allow users to upload new X-ray images for prediction.

---

# 🧠 Model Used

## VGG16

The project uses **VGG16**, a convolutional neural network architecture commonly used for image classification and feature extraction.

The model accepts images with the following input shape:

```text
224 × 224 × 3
🏁 Conclusion

The Pneumonia Detection from Chest X-Ray Images using VGG16 project successfully demonstrates the use of deep learning and transfer learning for medical image classification. The VGG16 model was trained to classify chest X-ray images into two categories: NORMAL and PNEUMONIA.

The model achieved a 93.55% training accuracy, 95.69% validation accuracy, and 84.94% test accuracy. The trained model was saved as pneumonia_vgg16.keras and integrated into a Streamlit web application.

The application allows users to upload a chest X-ray image, preprocesses it automatically, and displays the predicted result along with the model's confidence score. This project provides practical experience in dataset preparation, image preprocessing, transfer learning, model training, evaluation, and web deployment.

Overall, the project demonstrates how VGG16 can be applied to chest X-ray image classification. However, the system is intended only for educational purposes and should not be considered a replacement for professional medical diagnosis.
