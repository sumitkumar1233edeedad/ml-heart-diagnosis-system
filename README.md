# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

This project builds a Machine Learning classification system to predict the presence of heart disease using clinical and medical attributes. The objective is to assist in early risk detection and support healthcare decision-making using data-driven insights.

Multiple machine learning models are implemented and compared to determine the best-performing algorithm.

---

## 🎯 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early identification of cardiovascular risk can significantly improve treatment outcomes.

**Target Variable**

- 1 → Presence of Heart Disease  
- 0 → Absence of Heart Disease  

---

## 📂 Dataset Information

Dataset Source:

- UCI Machine Learning Repository – Heart Disease Dataset  
- Also available on Kaggle  

### Features Included

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate Achieved  
- Exercise-Induced Angina  
- ST Depression  
- Number of Major Vessels  
- Thalassemia  

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 🤖 Machine Learning Models Implemented

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

---

## 📊 Model Evaluation Metrics

- Confusion Matrix  
- Accuracy Score  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Score  
- Feature Importance  

---

## 🔄 Project Workflow

1. Data Collection  
2. Data Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Train-Test Split  
6. Model Training  
7. Model Evaluation  
8. Model Comparison  

---

## 📁 Project Structure

heart-disease-prediction-ml/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── images/
│   └── roc_curve.png
│
├── requirements.txt
├── app.py
└── README.md

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/heart-disease-prediction-ml.git  
cd ml-heart-diagnosis-system

### 2️⃣ Create Virtual Environment (Optional)

python -m venv venv  
venv\Scripts\activate  (Windows)  
source venv/bin/activate  (Mac/Linux)  

### 3️⃣ Install Dependencies

pip install -r requirements.txt  

### 4️⃣ Run Notebook

jupyter notebook  

---

## 📈 Expected Results

- Random Forest typically achieves highest accuracy  
- ROC Curve shows strong classification performance  
- Feature importance highlights key medical predictors  

---

## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Cross-validation  
- Deployment using Streamlit or Flask  
- Model explainability using SHAP  
- Real-time web integration  

---

## 🏥 Real-World Impact

This project demonstrates how Machine Learning can support early diagnosis of cardiovascular disease and assist healthcare professionals in preventive treatment planning.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Your Name  
Aspiring Data Scientist | Machine Learning Enthusiast