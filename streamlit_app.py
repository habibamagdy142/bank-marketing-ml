
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and preprocessor
preprocessor, model = joblib.load("bank_marketing_model_compressed.pkl")

st.title("Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

st.header("Customer Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

job = st.selectbox(
    "Job",
    ["admin.", "blue-collar", "entrepreneur", "housemaid",
     "management", "retired", "self-employed", "services",
     "student", "technician", "unemployed", "unknown"]
)

marital = st.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.selectbox(
    "Default",
    ["no", "yes"]
)

balance = st.number_input(
    "Balance",
    value=0
)

housing = st.selectbox(
    "Housing Loan",
    ["no", "yes"]
)

loan = st.selectbox(
    "Personal Loan",
    ["no", "yes"]
)

contact = st.selectbox(
    "Contact",
    ["cellular", "telephone", "unknown"]
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.selectbox(
    "Month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

campaign = st.number_input(
    "Number of Contacts During This Campaign",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.number_input(
    "Number of Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"]
)

st.divider()

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    input_encoded = preprocessor.transform(input_data)

    prediction = model.predict(input_encoded)[0]

    if prediction == "yes":
        st.success("The customer is likely to subscribe.")
    else:
        st.info("The customer is unlikely to subscribe.")
