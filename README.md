AquaTN — Tamil Nadu Water Usage Optimizer
A full-stack web application that predicts and optimizes water demand across Tamil Nadu's 38 districts using a Random Forest ML model.
Project Structure
```
water_optimizer/
├── app.py               # Flask backend + API routes
├── model.py             # Random Forest training & prediction
├── data_generator.py    # Synthetic Tamil Nadu dataset generator
├── requirements.txt
├── templates/
│   └── index.html       # Single-page application
└── static/
    ├── css/style.css    # Ocean-themed dark UI
    └── js/main.js       # Chart.js visualizations + API calls
```
Setup & Run
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Run the app
```bash
python app.py
```
The app will:
Generate a 5,000-sample synthetic dataset (Tamil Nadu water usage)
Train the Random Forest model with RandomizedSearchCV
Save the model to `model_artifacts.pkl`
Start the Flask server on `http://localhost:5000`
> **Note:** First run takes ~60–90 seconds for model training. Subsequent runs load the saved model instantly.
Features
Section	Description
Dashboard	District-wise bar chart + seasonal polar chart + KPI cards
Predict	Form with 12 input parameters → predicted water demand (MCM) + recommendations
Feature Analysis	Horizontal bar + doughnut chart of Random Forest feature importances
Optimization	Full district table with efficiency tiers and actionable recommendations
Model Accuracy	Scatter plot of actual vs. predicted + RMSE, MAE, R², CV-R² metrics
ML Architecture
Model: `RandomForestRegressor` (scikit-learn)
Tuning: `RandomizedSearchCV` with 5-fold CV, 20 iterations
Features: 12 features (district, crop, season, rainfall, temperature, reservoir level, population density, irrigated area, groundwater depth, soil moisture, evapotranspiration, industrial usage)
Target: Water demand in MCM (Million Cubic Meters)
Why Random Forest?
Non-linear relationships — Water demand has complex crop × season × rainfall interactions
Mixed feature types — Handles label-encoded categoricals natively
Outlier robustness — Bootstrap aggregation dampens extreme drought/flood year effects
Feature importance — Built-in Gini impurity importance for interpretability
No distributional assumptions — Agricultural data is often skewed; RF is assumption-free
Dataset
Synthetic dataset generated to reflect Tamil Nadu's agro-climatic reality:
38 districts with realistic population densities and rainfall patterns
16 crop types (paddy, sugarcane, cotton, millets, vegetables, etc.)
4 seasons (Kharif, Rabi, Zaid, Year-round)
Water demand computed from agricultural + domestic + industrial components
Northeast/Southwest monsoon rainfall patterns encoded per district
