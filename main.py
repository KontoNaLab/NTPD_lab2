from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

clf = joblib.load("iris_model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(input_data: IrisInput):
    features = np.array([[input_data.sepal_length, input_data.sepal_width, 
                          input_data.petal_length, input_data.petal_width]])
    prediction = clf.predict(features)
    probabilities = clf.predict_proba(features).tolist()
    return {"predicted_class": int(prediction[0]), "probabilities": probabilities}