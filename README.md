# Bank Marketing Prediction

A Machine Learning classification project that predicts whether a bank customer will subscribe to a term deposit based on customer and campaign-related information.

## Project Overview

The goal of this project is to build and evaluate machine learning models for predicting customer subscription to a term deposit.

The project includes data preprocessing, exploratory data analysis, model training, evaluation, model optimization, and deployment.

## Dataset

The project uses the **Bank Marketing Dataset**.

The dataset contains customer demographic information and details about previous marketing campaigns.

### Target Variable

* `y = yes` → The customer subscribed to a term deposit.
* `y = no` → The customer did not subscribe.

## Machine Learning Models

The following classification models were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Linear SVM
* Optimized Random Forest
* Optimized Gradient Boosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Deep Learning

A small neural network experiment was also included as an additional experiment.

It was not used as the final model because the traditional machine learning models provided better performance for this project.

## Final Model

The final selected model is the optimized model saved as:

`bank_marketing_model_final.pkl`

The model and preprocessing pipeline are saved together using Joblib.

## Deployment

The trained model was deployed using:

* **FastAPI** for the prediction API
* **Streamlit** for the interactive user interface

The application allows users to enter customer information and receive a prediction of whether the customer is likely to subscribe to a term deposit.

## Project Structure

```text
bank-marketing-ml/
│
├── bank_marketing_model.ipynb
├── app.py
├── streamlit_app.py
├── requirements.txt
└── bank_marketing_model_final.pkl
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI application:

```bash
uvicorn app:app --reload
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* FastAPI
* Streamlit
## Live Demo
Try the deployed application here:

[https://bank-marketing-ml-amjna8thmqp6kbhamqund2.streamlit.app/](https://bank-marketing-ml-amjna8thmqp6kbhamqund2.streamlit.app/)
* Joblib
* GitHub
