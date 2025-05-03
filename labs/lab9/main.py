from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
pipeline = joblib.load("reddit_model_pipeline.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    text = np.array([data.text])
    proba = pipeline.predict_proba(text).tolist()
    return {"probabilities": proba}

@app.get("/")
def read_root():
    return {"message": "Reddit Classifier is up"}