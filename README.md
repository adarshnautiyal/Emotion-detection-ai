# 😃 Real-Time Facial Emotion Detection using Deep Learning

A real-time AI system that detects human facial emotions from webcam input using **Deep Learning (CNN + MobileNetV2)** and **OpenCV face detection**.

---

## 🚀 Demo

📷 The system captures live webcam video and predicts emotions in real-time:

- 😡 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😄 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

---

## 🧠 Features

- 🎥 Real-time webcam emotion detection  
- 👤 Face detection using OpenCV DNN / Haar Cascade  
- 🤖 CNN-based deep learning model (MobileNetV2)  
- 🔄 Prediction smoothing for stable output  
- ⚡ Fast inference on CPU  
- 📊 Multi-class emotion classification  

---

## 🛠️ Tech Stack

- Python 🐍  
- TensorFlow / Keras 🤖  
- OpenCV 👁️  
- NumPy  
- MobileNetV2 (Transfer Learning)  

---

## 📁 Dataset

- FER-style facial emotion dataset  
- 7 emotion categories  
- Preprocessed with data augmentation:
  - Rotation
  - Zoom
  - Horizontal flip

---

## 🧠 Model Architecture

- Base Model: **MobileNetV2 (pretrained on ImageNet)**
- GlobalAveragePooling2D
- Dense Layers (128 units)
- Dropout (0.3)
- Softmax output layer (7 classes)

---

## 📊 Performance

- Training Accuracy: ~57%  
- Validation Accuracy: ~55%  
- Improved real-time stability using smoothing technique  

---

## 📂 Project Structure
