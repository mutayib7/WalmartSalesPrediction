from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app=FastAPI()
model=joblib.load('walmart_xgb_model.pkl')
features=['month', 'year', 'lag_1','lag_2', 'lag_3', 'lag_4','lag_8','lag_12','rolling_mean_4','rolling_mean_8','rolling_mean_12','IsHoliday']
class SalesInput(BaseModel):
    month: int 
    year: int 
    lag_1: float
    lag_2: float 
    lag_3: float 
    lag_4: float
    lag_8: float
    lag_12: float
    rolling_mean_4: float
    rolling_mean_8: float
    rolling_mean_12: float
    IsHoliday: int

@app.post("/predict")
def predict(data: SalesInput):
    df=pd.DataFrame([data.dict()])
    df=df[features]
    prediction=model.predict(df)[0]
    return {"predicted_sales": float(prediction)}

@app.get("/")
def root():
    return {"status":"API is running"}
