# Credit Scoring Model

## Project Overview
This project uses machine learning to estimate whether a loan applicant may default on a loan. It includes a Streamlit web app where users can enter applicant details and view a model prediction.

This is an educational project demo, not a real lending decision or financial advice.

## Features
- Loads and prepares a credit risk dataset.
- Trains a Logistic Regression classification model.
- Evaluates the model using accuracy, a classification report, and a confusion matrix.
- Saves the trained model for use in the web app.
- Provides a Streamlit interface for entering applicant details and viewing a prediction.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

## Project Files
- `main.py` — loads data, trains and evaluates the model, and saves the model.
- `app.py` — runs the Streamlit demo.
- `credit_risk_dataset.csv` — dataset used for training.
- `credit_scoring_model.pkl` — saved trained model.
- `confusion_matrix.png` — visualization of model evaluation.
- `requirements.txt` — Python package dependencies.

## Model Evaluation
The current test run reported an accuracy of approximately 86.02%.

The model's performance can vary with the data split and dataset. Check the classification report and confusion matrix to understand how it performs for each class, especially the Default class.

## How to Run Locally

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.
4. Install the required packages:

   ```bash
   pip install -r requirements.txt