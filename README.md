
# 🚗 Car Price Prediction App

### *AI-Powered Web App Built with Streamlit & Random Forest Regression*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML Model-orange?logo=scikitlearn">
  <img src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## 📌 Overview

This project is a **machine-learning powered web application** that predicts the **resale price of a car** based on its specifications.
The model is built using **Random Forest Regression**, and the user interface is crafted with **Streamlit** for a clean and interactive experience.

Users simply enter car details, and the app instantly provides:

✔ Estimated selling price
✔ Price comparison visualization
✔ Actual vs Predicted ML diagnostics
✔ Residual distribution
✔ Smart interpretation + actionable tips

---

## 🚀 Live Demo (If hosting on Streamlit Cloud)

https://ajaniyasri-car-price-prediction-app-e6dhxh.streamlit.app

---

## 🖼️ UI Preview

> Add screenshots after pushing project

```
assets/
 ├── homepage.png
 ├── prediction.png
 ├── diagnostics.png
```

You can embed them like:

```md
![Home](assets/homepage.png)
![Prediction](assets/prediction.png)
```

---

## ⭐ Features

### 🔍 **Car Price Estimation**

* Predicts resale price (in lakhs)
* Inputs include:

  * Present price
  * Kilometers driven
  * Previous owners
  * Manufacturing year
  * Fuel type
  * Seller type
  * Transmission type

### 📊 **Machine Learning Diagnostics**

* Actual vs Predicted scatter plot
* Residuals distribution
* R², RMSE, MAE metrics

### 📈 **Visual Insights**

* Present vs predicted price bar chart
* Clean and compact layout
* Modern UI with reduced diagram size

### 💡 **Smart Tips**

* Interprets prediction
* Gives suggestions to improve resale value

---

## 🧠 Machine Learning Model

* Algorithm → **Random Forest Regressor**
* Training File → `random_forest_regression_model.pkl`
* Preprocessing:

  * Fuel Type: One-hot encoded
  * Seller Type: Binary
  * Transmission: Binary
  * No_Year created from Manufacturing Year

To maintain compatibility, your dataset must follow:

```
Present_Price
Kms_Driven
Owner
No_Year
Fuel_Type_Diesel
Fuel_Type_Petrol
Seller_Type_Individual
Transmission_Manual
Selling_Price
```

---

## 🛠️ Tech Stack

| Component     | Technology                     |
| ------------- | ------------------------------ |
| ML Model      | Random Forest Regression       |
| UI Framework  | Streamlit                      |
| Visualization | Matplotlib, Seaborn            |
| Language      | Python                         |
| Dataset       | Car price dataset (supervised) |

---

## 📥 Installation & Run Locally

### 🔧 **1. Clone the repository**

```bash
git clone https://github.com/AjaniyaSri/Car-price-prediction.git
cd car-price-prediction
```

### 📦 **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### ▶️ **3. Run the Streamlit app**

```bash
streamlit run app.py
```


## 📂 Project Structure

```
car-price-prediction/
│
├── app.py
├── random_forest_regression_model.pkl
├── car data.csv
├── requirements.txt
├── assets/
│    ├── homepage.png
│    ├── prediction.png
│    └── diagnostics.png
└── README.md
```


## 🧪 Model Performance Summary

| Metric   | Value                   |
| -------- | ----------------------- |
| R² Score | ~0.90+ (example)        |
| RMSE     | varies based on dataset |
| MAE      | varies based on dataset |



## 🤝 Contributing

Contributions are welcome! Feel free to open:

* Issues
* Pull requests
* Feature suggestions


## 👨‍💻 Author

**Ajani**
🚀 Student | ML Developer | Data Enthusiast


[LinkedIn](www.linkedin.com/in/ajaniyakamalanathan) | [GitHub](https://github.com/AjaniyaSri) 


