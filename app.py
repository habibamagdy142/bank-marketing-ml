
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and preprocessor
preprocessor, model = joblib.load("bank_marketing_model_final.pkl")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Bank Marketing Prediction API is running"}


class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.dict()])

    input_encoded = preprocessor.transform(input_data)

    prediction = model.predict(input_encoded)[0]

    return {
        "prediction": prediction
    }
