# Wealth Coop Flask Banking Prediction System

## 📌 Overview
This project is a **Flask-based backend API** for a smart banking prediction system. It provides ML-powered predictions for:

- ✅ **Loan Default Approval**
- 🔁 **Loan Repayment Risk**
- 💰 **Loan Amount Forecasting**

Built for **Wealth Cooperative Bank**, it helps automate and optimize decision-making for loan management.

---

## 🚀 Features

### 🔍 Endpoints

- `POST /predict-default` – Predict loan default approval (Approve / Reject)  
- `POST /predict-repayment` – Predict repayment risk (High / Medium / Low)  
- `POST /forecast-loan-amount` – Predict the ideal loan amount based on user profile

### 📊 ML Models

- Trained using `scikit-learn` pipelines  
- Models and preprocessors saved as `.pkl` files for fast loading

---

## 📂 Project Structure

```
deploy-project/
├── data-files/                  # CSV datasets
│   ├── Loan-Default.csv         # Loan Default Dataset
|   ├── Loan-forcast.csv         # Loan Forcast Dataset
|   ├── Loan-repayment.csv       # Loan repayment Dataset
├── data-visualization/          # Notebooks for data exploration
|   ├── default_visualization.ipynb    # Jupyter Loan default visualization Notebook
|   ├── forcast_visualization.ipynb    # Jupyter Loan forcast visualization Notebook
|   ├── repay_visualization.ipynb      # Jupyter Loan repayment visualization Notebook
├── default-model/           # Default prediction model + preprocessor
|   ├── model.pkl               # load default model
|   ├── preprocessor.pkl        # load default preprocessor
├── loan-forcast-model/      # Loan forecasting model + preprocessor
|   ├── model.pkl               # load forcast model
|   ├── preprocessor.pkl        # load forcast preprocessor
├── repayment-model/         # Repayment prediction model + preprocessor
|   ├── model.pkl               # load repayment model
|   ├── preprocessor.pkl        # load repayment preprocessor
├── templates/               # Flask HTML templates (optional)
|   ├── index.html               # load default model
├── tests/                   # Testing scripts for APIs
|   ├── app.py                # load default train
|   ├── app2.py               # load forcast train
|   ├── app3.py               # load repayment train
|   ├── test.py               # example model trail file
├── train-files/             # Model training scripts and notebooks
|   ├── app.py                # load default train
|   ├── app2.py               # load forcast train
|   ├── app3.py               # load repayment train
├── requirements.txt         # Backend dependencies
└── flaskserver.py           # Main Flask API backend


```

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/wealth-coop-flask-app.git
cd wealth-coop-flask-app
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Train Files

```bash
python default.py
python forcast.py
python repay.py
# you maybe required to cd into the directory
```

---

## ▶️ Run the Server

```bash
python flaskserver.py
```

> The API will be available at: [http://localhost:5000](http://localhost:5000)

---

## 📩 Example Request – Loan Default Prediction

**Endpoint:** `POST /predict-default`

### ✅ Request JSON

```json
{
  "loan_limit": "Yes",
  "Gender": "Male",
  "loan_type": "Personal",
  "business_or_commercial": "No",
  "loan_amount": 50000,
  "rate_of_interest": 12.5,
  "Interest_rate_spread": 2.5,
  "income": 85000,
  "credit_type": "Good",
  "Credit_Score": 720,
  "age": 35
}
```

### 🔁 Response

```json
{
  "prediction": "Approve"
}
```

---

## 📌 Notes

- Ensure all models and their respective preprocessors (`.pkl` files) are stored in the correct folders.
- This is a **backend-only system**, but can be integrated with any React frontend or mobile banking app.

---

## 👨‍💻 Authors

Developed by Dasun Wijekumara

