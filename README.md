# Walmart Sales Prediction

## Problem Statement
Predict Weekly sales for Walmart stores based on historical sales data,store information and external factor holiday.

## Live Demo on Render
- **[Try the API here:](https://walmart-sales-prediction-api.onrender.com/docs)**

## Dataset 
- Walmart Store Sales dataset
- 421570 samples, 5 features
- Features: Store, Dept, Date, Weekly_Sales, IsHoliday
- Target Variable: Weekly_Sales 

## Approach
- Performed EDA to identify holiday effects and seasonal trends
- Engineered Features: Month, Year, lag_1, lag_2, lag_3, lag_4, lag_8, lag_12, rolling_mean_4, rolling_mean_8, rolling_mean_12
- Models trained and compared:
  ○ Random Forest Regressor
  ○ XGBoost Regressor
  ○ Prophet Model
  ○ LightGBM
- Evaluation Metric: RMSE

## Results
- RMSE Results for different models:
  ○ Random Forest Regressor: 2606208
  ○ XGBoost Regressor: 1879051
  ○ Prophet Model: 4511039
  ○ LightGBM: 2685704
- Best Model: XGBoost Regressor
- Holiday Weeks showed the highest sales spikes
- Lag features and rolling averages were the most impactful predictors
  
## API Deployment
- A FastAPI endpoint serves the trained XGBoost model for real-time sales predictions.

### Run API locally
- uvicorn app:app --reload
- Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI testing.

### Notes & Limitations
- The `/predict` endpoint currently requires precomputed lag and rolling-mean features as input

## How to Run Notebook
- pip install -r requirements.txt
- Open SalesPrediction.ipynb and run all cells.

## Run with Docker
- docker build -t walmart-sales-api .
- docker run -d -p 8000:8000 walmart-sales-api
- Visit `http://127.0.0.1:8000/docs` 

## Tech Stack
- Python, scikit-learn, XGBoost, Prophet, LightGBM, numpy, pandas, matplotlib, seaborn, fastAPI, uvicorn, pydantic, joblib
