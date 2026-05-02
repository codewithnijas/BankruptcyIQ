# 📉 BankruptcyIQ — Startup Risk Intelligence Platform

> *Predict startup financial distress using machine learning and financial indicators.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square\&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square\&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🧠 Overview

**BankruptcyIQ** is an interactive machine learning application that predicts the probability of startup bankruptcy based on financial, operational, and business factors.

The app provides a **fintech-style dashboard** where users can input startup data, evaluate risk in real-time, and compare multiple machine learning models.

Built as a portfolio project to demonstrate **end-to-end ML system design with a production-style UI**.

---

## ✨ Features

| Feature                           | Description                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| 🎯 **Startup Risk Prediction**    | Interactive form with real-time probability and risk classification        |
| 📊 **Model Evaluation Dashboard** | Confusion matrix, ROC curve, Precision-Recall curve, classification report |
| 🔍 **Feature Analysis**           | Feature importance (for supported models) + correlation heatmap            |
| ⚖️ **Model Benchmarking**         | Compare 6 ML models across Accuracy, ROC-AUC, and Average Precision        |
| 🎨 **Fintech UI**                 | Custom dark theme with clean layout and interactive visualizations         |

---

## 🏗️ Architecture

```
BankruptcyIQ/
├── app.py                  # Main Streamlit application
├── startup.csv             # Dataset
├── requirements.txt        # Dependencies
├── README.md               # Documentation
└── .gitignore
```

---

## 🤖 Models

Six machine learning models are implemented and evaluated:

* **Random Forest** *(default)*
* **Gradient Boosting**
* **Logistic Regression**
* **Decision Tree**
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**

Each model is trained using a consistent preprocessing pipeline and evaluated on a hold-out test set.

---

## ⚙️ Feature Engineering

The model uses both raw and derived financial features:

| Feature         | Description                                |
| --------------- | ------------------------------------------ |
| Revenue_to_Burn | Measures efficiency of revenue vs expenses |
| Debt_to_Funding | Indicates financial leverage risk          |
| Cost_to_Revenue | Captures operational efficiency            |
| Funding_per_Emp | Capital allocation per employee            |
| Revenue_per_Emp | Productivity per employee                  |

---

## 📊 Model Evaluation

The application includes:

* **Confusion Matrix** — classification performance
* **ROC Curve (AUC)** — ability to separate classes
* **Precision-Recall Curve** — performance on imbalanced data
* **Classification Report** — precision, recall, F1-score

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/BankruptcyIQ
cd BankruptcyIQ

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📁 Dataset

The app expects a dataset (`startup.csv`) with the following columns:

```
Funding_Amount_MUSD, Revenue_MUSD, Profit_Margin,
Burn_Rate_MUSD, Employees, Years_Active,
Customer_Growth_Rate, Debt_Level_MUSD,
Operational_Cost_MUSD, Market_Competition,
Founder_Experience, Industry, Region, Bankrupt
```

* `Bankrupt` is the target variable (0 = Healthy, 1 = Bankrupt)

---

## 🧩 Tech Stack

| Layer         | Tools               |
| ------------- | ------------------- |
| UI            | Streamlit           |
| ML            | scikit-learn        |
| Data          | pandas, numpy       |
| Visualization | matplotlib, seaborn |

---

## 🎯 Use Cases

* Startup risk analysis
* Investment decision support
* Financial health monitoring
* Machine learning portfolio projects

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio use only**.
Predictions should not be used for real financial decision-making.

---

## 👨‍💻 Author

**Muhammed Nijas**


---

## ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share it

---

## 📜 License

This project is licensed under the MIT License.
