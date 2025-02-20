# 🎗️ BRAIN TUMOR DETECTION USING DEEP LEARNING

## 📌 OVERVIEW

Brain tumors are among the most critical medical conditions that require early and accurate detection for effective treatment. This project leverages Deep Learning and Computer Vision techniques to classify brain MRI scans into tumorous (Yes) and non-tumorous (No) categories. A Convolutional Neural Network (CNN) model is trained on MRI images to detect brain tumors with high accuracy.


## 🚀 FEATURES

✔️ Deep Learning-based classification using a CNN model.

✔️ Flask-powered Web Application for real-time predictions.

✔️ Trained on MRI scan datasets (tumorous vs non-tumorous images).

✔️ Real-time inference using OpenCV and TensorFlow.

✔️ Well-structured and modular codebase for easy scalability.


## 🔬 ALGORITHM & MODEL DETAILS

This project utilizes a Convolutional Neural Network (CNN) with the following architecture:

**Input Layer:** 64x64 RGB MRI images.

**Feature Extraction:**

**Conv2D Layers:** Multiple convolutional layers with ReLU activation.

**MaxPooling Layers:** To reduce spatial dimensions and computational load.

**Fully Connected Layers:**

**Flatten Layer:** Converts 2D feature maps into a 1D vector.

**Dense Layers:** Uses ReLU and softmax activation for classification.

**Output Layer:** 2 classes (Tumor / No Tumor).

**Optimizer:** Adam Optimizer.

**Loss Function:** Categorical Crossentropy.

Model is trained for 10 epochs with a batch size of 16.


## 🛠️ TECH STACK & DEPENDENCIES

**This project is built using:**

🟠 Python 3.x

🟡 TensorFlow/Keras – Deep Learning Framework

🔵 OpenCV – Image Processing

🟢 NumPy – Numerical Computation

🔴 Flask – Web Framework

🟣 PIL (Pillow) – Image Processing

**Install all dependencies using:**

pip install tensorflow keras numpy flask opencv-python pillow

## 🚀 HOW TO RUN

🔹 **1️⃣ Clone the Repository**

git clone https://github.com/yourusername/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

🔹 **2️⃣ Train the Model (Optional)**

If you want to retrain the model, run:

python mainTrain.py

🔹 **3️⃣ Start the Flask Application**

Run the following command to start the web app:

python app.py

The application will be available at http://127.0.0.1:5000/.

🔹 **4️⃣ Upload an Image for Prediction**

Open the web app.

Upload an MRI scan.

The model will predict whether the image contains a tumor or not.


## 📌 FUTURE ENHANCEMENTS

🚀 Improve Model Accuracy: Train with a larger dataset and hyperparameter tuning.

☁️ Deploy to Cloud: Host the Flask application on AWS/GCP for real-world usage.

🔍 Integrate Explainability: Use Grad-CAM for model interpretability.

🎨 Enhance UI: Make the frontend more interactive and visually appealing.


✨ AUTHOR

👤 Abhinav Sriharsha 📧 abhinav932002@gmail.com

💡 If you find this project useful, consider giving it a ⭐ on GitHub!

