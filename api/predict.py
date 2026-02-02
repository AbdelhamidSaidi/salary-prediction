from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'salary_model.pkl')

model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


class Features(BaseModel):
    years: float
    jobrate: float


@app.get('/')
def root():
    return {'status': 'salary prediction API', 'endpoint': '/api/predict'}


@app.post('/api/predict')
def predict(features: Features):
    m = get_model()
    X = np.array([[features.years, features.jobrate]])
    pred = m.predict(X)
    return {'prediction': float(pred[0])}
