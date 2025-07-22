from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

class PatientData(BaseModel):
    age: float
    sex: str #M or F 
    bmi: float
    bp: float   # blood pressure
    s1: float   # serum measurement 1
    s2: float   # serum measurement 2  
    s3: float   # serum measurement 3
    s4: float   # serum measurement 4
    s5: float   # serum measurement 5
    s6: float   # serum measurement 6
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 48,
                "sex": "M",
                "bmi": 25.3,
                "bp": 90,
                "s1": 180,
                "s2": 100,
                "s3": 90,
                "s4": 4,
                "s5": 4.2,
                "s6": 90
            }
        }

with open("models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0"
)

@app.post("/predict")
def predict_progression(patient: PatientData):
    sex_map = {'M':1, 'F':0}
    sex_val = sex_map.get(patient.sex.upper(), 0)


    features = np.array([[
        patient.age, sex_val, patient.bmi, patient.bp,
        patient.s1, patient.s2, patient.s3, patient.s4,
        patient.s5, patient.s6
    ]])

    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)[0]
    
    return {
        "predicted_progression_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }
 
def get_interpretation(score):
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"
    
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "diabetes_progression_v1"}