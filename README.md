# 🍎 Pomegranate Disease Detector

A deep learning–based web application that detects and classifies **pomegranate leaf diseases** using a **Convolutional Neural Network (CNN)**.  
This project is built using **Flask**, **TensorFlow**, **NumPy**, and **OpenCV**, and provides an intuitive web interface for users to upload leaf images and receive instant disease predictions.

---

## 📊 Dataset

The dataset used for training is available on **Kaggle**:

🔗 [Pomegranate Diseases Dataset on Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/sujaykapadnis/pomegranate-fruit-diseases-dataset))  
*(Replace this with your actual dataset link if available.)*

### 📁 After Downloading:
1. Extract the dataset folder into your project directory.  
2. The folder structure should look like this:

Install All Dependencies
pip install -r requirements.txt

Prepare the Dataset
python prepare_dataset.py

Train the CNN Model
python train_model.py

Run the Flask Web App
python app.py
