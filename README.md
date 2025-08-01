# 🌾 AgriVision – Crop Disease Detection System

AgriVision is a web-based application that leverages deep learning (CNN) to detect crop diseases from uploaded images. Built with Flask, it provides farmers and researchers with a fast, accessible way to identify and manage plant diseases.

---

## 🚀 Features

- 📷 Upload crop images and detect diseases using a trained CNN model
- 👤 User Authentication (Sign In / Sign Up)
- 🖼️ Clean UI using HTML templates
- 🔒 Session-based login
- 🌱 Fast and lightweight backend (Flask)

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3 (with Jinja2 templating)
- **Backend:** Flask (Python)
- **Model:** Convolutional Neural Network (Keras/TensorFlow)
- **Environment:** Python + Virtualenv

---

## 📁 Project Structure

CropDiseaseApp/
├── backend/
│ ├── app.py # Main Flask app
│ ├── templates/ # HTML templates
│ │ ├── index.html
│ │ ├── signin.html
│ │ └── signup.html
├── .venv/ # Virtual environment (ignored)
├── .gitignore
├── requirements.txt
└── README.md
