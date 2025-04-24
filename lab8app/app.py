from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Wine Classifier",
    description="Classify wines using a model from MLFlow",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a wine classifier model'}

class request_body(BaseModel):
    features : list

@app.on_event('startup')
def load_artifacts():
    global model
    # Load the model using joblib
    model = joblib.load("../labs/mlruns/1/f8f25fc974914493b515294194906d8c/artifacts/my_models/model.pkl")

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    X = data.features
    predictions = model.predict(X)
    return {'Predictions': predictions.tolist()} 