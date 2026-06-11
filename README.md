# Loan Approval Prediction System

A Machine Learning web application that predicts whether a loan application is likely to be approved or rejected based on applicant details. The project uses a Random Forest Classifier trained on the Loan Prediction dataset and is deployed using Flask and Render.

## Live Demo

🔗 Live Application: https://loan-approval-predictor-tbkv.onrender.com/

---

## Project Overview

This project was developed to automate loan approval prediction using historical applicant data. The model analyzes applicant information such as income, loan amount, credit history, education, and property area to determine loan eligibility.

The project demonstrates:

* Data Preprocessing
* Feature Engineering
* Handling Class Imbalance using SMOTE
* Machine Learning Model Comparison
* Random Forest Classification
* Model Deployment using Flask
* Cloud Deployment using Render

---

## Features

* Predict loan approval status instantly
* User-friendly web interface
* Random Forest based prediction model
* Automated feature engineering
* Responsive design
* Real-time prediction results

---

## Machine Learning Workflow

### Data Preprocessing

* Missing value treatment
* Label Encoding for categorical variables
* Feature Scaling using StandardScaler
* Feature Engineering

### Feature Engineering

Created additional features:

* TotalIncome = ApplicantIncome + CoapplicantIncome
* ApplicantIncome_Log
* LoanAmount_Log

### Class Imbalance Handling

* SMOTE (Synthetic Minority Oversampling Technique)

### Models Compared

* Logistic Regression
* Random Forest Classifier

### Evaluation Metrics

* Precision
* Recall
* F1 Score
* ROC-AUC Score

---

## Model Performance

### Logistic Regression

* Precision: 0.874
* Recall: 0.894
* F1 Score: 0.884
* ROC-AUC: 0.793

### Random Forest

* Precision: 0.851
* Recall: 0.871
* F1 Score: 0.860
* ROC-AUC: 0.818

Random Forest was selected for deployment due to its strong ROC-AUC performance and ability to capture complex relationships within the data.

---

## Business Insights

Key factors influencing loan approval:

* Credit History
* Applicant Income
* Total Income
* Loan Amount
* Property Area

Applicants with strong credit history and stable income have a significantly higher probability of loan approval.

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn

### Visualization

* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS
* Bootstrap

### Deployment

* Render

---

## Project Structure

```text
LoanApprovalApp/
│
├── app.py
├── requirements.txt
├── Procfile
├── random_forest_model.pkl
├── scaler.pkl
│
├── notebooks/
│   └── Loan_Approval_Prediction.ipynb
│
├── data/
│   └── final_cleaned_dataset.csv
│
├── templates/
│
├── static/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd LoanApprovalApp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Deployment

The application is deployed on Render.

Live URL:

https://loan-approval-predictor-tbkv.onrender.com/

---

## Future Improvements

* XGBoost implementation
* Explainable AI using SHAP
* User authentication
* Loan eligibility score
* REST API integration
* Advanced analytics dashboard

---

## Author

Afzal

B.Tech Student | Machine Learning Enthusiast | Aspiring Software Engineer

---

## License

This project is developed for educational and portfolio purposes.
