# Walmart Sales Prediction

## Problem Statement
Predict Weekly sales for Walmart stores based on historical sales data,store information and external factor holiday.

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
 
## How to Run
pip install -r requirements.txt
Open SalesPrediction.ipynb and run all cells.

## Tech Stack
Python, scikit-learn, XGBoost, Prophet, LightGBM, numpy, pandas, matplotlib, seaborn
