"""
Loan Approval Prediction – Flask Application
Uses a pre-trained Random Forest model to predict loan approval.
"""

import logging
import os
import pickle
import warnings

import numpy as np
from flask import Flask, render_template, request

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths – resolve relative to this file so it works in any working directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Load model, scaler, and dataset once at startup
# ---------------------------------------------------------------------------
def load_model():
    """Load the trained Random Forest model from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def load_scaler():
    """Load the fitted StandardScaler from disk."""
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler


model = load_model()
scaler = load_scaler()

# Feature order expected by the scaler/model
FEATURE_COLS = list(scaler.feature_names_in_)

# Log startup information
app.logger.info("Model loaded: %s", type(model).__name__)
app.logger.info("Features: %s", FEATURE_COLS)

# Metrics computation removed


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------
ENCODING_MAP = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {"1": 1.0, "0": 0.0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

REQUIRED_FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


def validate_form_data(form_data: dict) -> list[str]:
    """
    Validate that all required fields are present and contain
    acceptable values.  Returns a list of error messages (empty
    if valid).
    """
    errors: list[str] = []

    # Check all required fields are present and non-empty
    for field in REQUIRED_FIELDS:
        if field not in form_data or not form_data[field].strip():
            errors.append(f"Missing required field: {field.replace('_', ' ')}")

    if errors:
        return errors  # bail early – can't check values of missing fields

    # Validate categorical fields against the encoding map
    for field, valid_values in ENCODING_MAP.items():
        val = form_data.get(field, "")
        if val not in valid_values:
            errors.append(
                f"Invalid value '{val}' for {field.replace('_', ' ')}. "
                f"Expected one of: {', '.join(valid_values.keys())}"
            )

    # Validate numeric fields
    for field in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"):
        try:
            value = float(form_data[field])
            if value < 0:
                errors.append(f"{field.replace('_', ' ')} cannot be negative.")
        except (ValueError, TypeError):
            errors.append(f"{field.replace('_', ' ')} must be a valid number.")

    # Dependents must be an integer 0-3
    try:
        dep = int(form_data["Dependents"])
        if dep < 0 or dep > 3:
            errors.append("Dependents must be between 0 and 3.")
    except (ValueError, TypeError):
        errors.append("Dependents must be a valid integer.")

    return errors


def preprocess_input(form_data: dict) -> np.ndarray:
    """
    Convert raw form data into the 14-feature vector required by the model.
    Applies label-encoding, feature engineering (log transforms, total income),
    and standard-scaling.
    """
    gender = ENCODING_MAP["Gender"][form_data["Gender"]]
    married = ENCODING_MAP["Married"][form_data["Married"]]
    dependents = int(form_data["Dependents"])
    education = ENCODING_MAP["Education"][form_data["Education"]]
    self_employed = ENCODING_MAP["Self_Employed"][form_data["Self_Employed"]]
    applicant_income = float(form_data["ApplicantIncome"])
    coapplicant_income = float(form_data["CoapplicantIncome"])
    loan_amount = float(form_data["LoanAmount"])
    loan_amount_term = float(form_data["Loan_Amount_Term"])
    credit_history = ENCODING_MAP["Credit_History"][form_data["Credit_History"]]
    property_area = ENCODING_MAP["Property_Area"][form_data["Property_Area"]]

    # Engineered features
    total_income = applicant_income + coapplicant_income
    applicant_income_log = np.log1p(applicant_income)
    loan_amount_log = np.log1p(loan_amount)

    raw = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area,
        total_income,
        applicant_income_log,
        loan_amount_log,
    ]])

    scaled = scaler.transform(raw)
    return scaled


def predict_loan(scaled_input: np.ndarray) -> dict:
    """Run prediction and return label + probabilities."""
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    return {
        "prediction": prediction,
        "label": "Approved" if prediction == 1 else "Rejected",
        "approval_prob": round(float(probabilities[1]) * 100, 2),
        "rejection_prob": round(float(probabilities[0]) * 100, 2),
    }





# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()

    # Validate inputs before processing
    validation_errors = validate_form_data(form)
    if validation_errors:
        return render_template(
            "result.html",
            error="Please fix the following: " + "; ".join(validation_errors),
        )

    try:
        scaled_input = preprocess_input(form)
        result = predict_loan(scaled_input)
        result["form"] = form
        app.logger.info(
            "Prediction: %s (approval=%.1f%%) | Income=%s, Loan=%s",
            result["label"],
            result["approval_prob"],
            form.get("ApplicantIncome"),
            form.get("LoanAmount"),
        )
        return render_template("result.html", result=result)
    except Exception:
        app.logger.exception("Prediction failed for input: %s", form)
        return render_template(
            "result.html",
            error="An unexpected error occurred while processing your request. "
                  "Please verify your inputs and try again.",
        )


# Dashboard route removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
