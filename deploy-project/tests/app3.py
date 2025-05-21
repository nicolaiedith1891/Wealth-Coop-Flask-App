from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model and preprocessor
with open("../loan-forcast-model/loan_model.pkl", "rb") as file:
    model = pickle.load(file)
    logger.debug("Loan amount model loaded successfully")

with open("../loan-forcast-model/loan_preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)
    logger.debug("Preprocessor loaded successfully")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define expected features based on your dataset
feature_names = [
    'customer_age', 'customer_income', 'home_ownership',
    'employment_duration', 'loan_intent', 'loan_grade',
    'loan_int_rate', 'term_years', 'historical_default',
    'cred_hist_length'
]

@app.route('/forcast-loan-amount', methods=['POST'])
def predict_loan_amount():
    try:
        # Log incoming request
        data = request.get_json()
        logger.debug(f"Received request data: {data}")

        # Validate and prepare input
        input_data = {}
        for feature in feature_names:
            value = data.get(feature)
            
            # Handle missing values with reasonable defaults
            if value is None:
                logger.warning(f"Missing feature: {feature}. Using default.")
                if feature in ['customer_age', 'customer_income', 'employment_duration', 
                              'term_years', 'cred_hist_length']:
                    value = 0  # Default for numerical
                elif feature == 'loan_int_rate':
                    value = 10.0  # Average interest rate
                elif feature == 'historical_default':
                    value = 'N'  # Assume no default history
                else:
                    value = 'UNKNOWN'  # For categoricals
                    
            input_data[feature] = [value]  # Wrap in list for DataFrame
        
        logger.debug(f"Processed input: {input_data}")

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Clean numerical fields (in case strings come through API)
        num_cols = ['customer_age', 'customer_income', 'employment_duration',
                   'loan_int_rate', 'term_years', 'cred_hist_length']
        for col in num_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Transform features
        transformed_input = preprocessor.transform(input_df)
        
        # Predict loan amount
        predicted_amount = model.predict(transformed_input)[0]
        predicted_amount = float(predicted_amount)  # Convert to native Python float
        
        logger.debug(f"Predicted loan amount: £{predicted_amount:,.2f}")

        return jsonify({
            "predicted_loan_amount": predicted_amount,
            "currency": "LKR",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == '__main__':
    logger.info("Starting loan amount prediction API")
    app.run(host='0.0.0.0', port=5000, debug=True)