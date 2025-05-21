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
with open("../repayment-model/model.pkl", "rb") as file:
    model = pickle.load(file)
    logger.debug("Repayment model loaded successfully")

with open("../repayment-model/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)
    logger.debug("Preprocessor loaded successfully")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define expected features (adjust based on your actual columns)
feature_names = [
    'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
    'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
]

@app.route('/predict-repayment', methods=['POST'])
def predict_repayment():
    try:
        # Log incoming request
        data = request.get_json()
        logger.debug(f"Received request data: {data}")

        # Validate and prepare input
        user_input = []
        for feature in feature_names:
            value = data.get(feature)
            if value is None:
                logger.warning(f"Missing feature: {feature}. Using default.")
                value = 0 if feature in ['credit.policy', 'delinq.2yrs', 'pub.rec', 'inq.last.6mths'] else 0.0
            user_input.append(value)
        
        logger.debug(f"Processed input: {user_input}")

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input], columns=feature_names)

        # Transform features
        transformed_input = preprocessor.transform(input_df)
        
        # Predict repayment percentage
        repayment_percentage = model.predict(transformed_input)[0]
        repayment_percentage = float(np.clip(repayment_percentage, 0, 1))  # Ensure between 0-1
        
        logger.debug(f"Predicted repayment percentage: {repayment_percentage:.2%}")

        return jsonify({
            "prediction": repayment_percentage,
            "risk_category": "Low" if repayment_percentage >= 0.9 else "Medium" if repayment_percentage >= 0.7 else "High"
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting loan repayment prediction API")
    app.run(host='0.0.0.0', port=5000, debug=True)