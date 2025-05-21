from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------- Load All Models and Preprocessors ----------------
# Loan Default Prediction
with open("default-model/model.pkl", "rb") as file:
    default_model = pickle.load(file)
with open("default-model/preprocessor.pkl", "rb") as file:
    default_preprocessor = pickle.load(file)

# Loan Repayment Prediction
with open("repayment-model/model.pkl", "rb") as file:
    repayment_model = pickle.load(file)
with open("repayment-model/preprocessor.pkl", "rb") as file:
    repayment_preprocessor = pickle.load(file)

# Loan Amount Forecast
with open("loan-forcast-model/loan_model.pkl", "rb") as file:
    forecast_model = pickle.load(file)
with open("loan-forcast-model/loan_preprocessor.pkl", "rb") as file:
    forecast_preprocessor = pickle.load(file)

# ---------------- Feature Names ----------------
default_feature_names = ['loan_limit', 'Gender', 'loan_type', 'business_or_commercial',
                         'loan_amount', 'rate_of_interest', 'Interest_rate_spread',
                         'income', 'credit_type', 'Credit_Score', 'age']

repayment_feature_names = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
                            'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
                            'inq.last.6mths', 'delinq.2yrs', 'pub.rec']

forecast_feature_names = ['customer_age', 'customer_income', 'home_ownership',
                           'employment_duration', 'loan_intent', 'loan_grade',
                           'loan_int_rate', 'term_years', 'historical_default',
                           'cred_hist_length']

# ---------------- Initialize Flask ----------------
app = Flask(__name__)
CORS(app)

# ---------------- API Endpoints ----------------

# Loan Default Prediction
@app.route('/predict-default', methods=['POST'])
def predict_default():
    try:
        data = request.get_json()
        logger.debug(f"Default prediction request data: {data}")

        user_input = [data.get(feature, 0) for feature in default_feature_names]

        # Convert selected fields to float
        num_indices = [4, 5, 6, 7, 9]
        for i in num_indices:
            user_input[i] = float(user_input[i])

        input_df = pd.DataFrame([user_input], columns=default_feature_names)

        transformed_input = default_preprocessor.transform(input_df)
        prediction = default_model.predict(transformed_input)

        result = "Approve" if prediction[0] == 1 else "Reject"

        return jsonify({"prediction": result})

    except Exception as e:
        logger.error(f"Default prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# Loan Repayment Prediction
@app.route('/predict-repayment', methods=['POST'])
def predict_repayment():
    try:
        data = request.get_json()
        logger.debug(f"Repayment prediction request data: {data}")

        user_input = []
        for feature in repayment_feature_names:
            value = data.get(feature, 0)
            user_input.append(value)

        input_df = pd.DataFrame([user_input], columns=repayment_feature_names)

        transformed_input = repayment_preprocessor.transform(input_df)
        repayment_percentage = repayment_model.predict(transformed_input)[0]
        repayment_percentage = float(np.clip(repayment_percentage, 0, 1))

        return jsonify({
            "prediction": repayment_percentage,
            "risk_category": "Low" if repayment_percentage >= 0.9 else "Medium" if repayment_percentage >= 0.7 else "High"
        })

    except Exception as e:
        logger.error(f"Repayment prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# Loan Amount Forecast
@app.route('/forecast-loan-amount', methods=['POST'])
def forecast_loan_amount():
    try:
        data = request.get_json()
        logger.debug(f"Loan forecast request data: {data}")

        input_data = {}
        for feature in forecast_feature_names:
            value = data.get(feature)
            if value is None:
                if feature in ['customer_age', 'customer_income', 'employment_duration', 'term_years', 'cred_hist_length']:
                    value = 0
                elif feature == 'loan_int_rate':
                    value = 10.0
                elif feature == 'historical_default':
                    value = 'N'
                else:
                    value = 'UNKNOWN'
            input_data[feature] = [value]

        input_df = pd.DataFrame(input_data)

        num_cols = ['customer_age', 'customer_income', 'employment_duration',
                    'loan_int_rate', 'term_years', 'cred_hist_length']
        for col in num_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        transformed_input = forecast_preprocessor.transform(input_df)
        predicted_amount = forecast_model.predict(transformed_input)[0]
        predicted_amount = float(predicted_amount)

        return jsonify({
            "predicted_loan_amount": predicted_amount,
            "currency": "LKR",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Loan forecast error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# ---------------- Run Server ----------------
if __name__ == '__main__':
    logger.info("Starting unified Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)