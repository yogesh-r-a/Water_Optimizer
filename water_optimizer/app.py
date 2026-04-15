"""
app.py
Flask backend for Tamil Nadu Water Usage Optimizer.
Handles model training, serving API endpoints, and HTML rendering.
"""

import os
import json
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from data_generator import generate_dataset, DISTRICTS, CROPS, SEASONS
from model import WaterDemandModel

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

MODEL_PATH = "model_artifacts.pkl"
DATA_PATH = "data/tn_water_usage.csv"

# Global instances — loaded once at startup
_model = WaterDemandModel()
_df = None          # Full dataset (pandas DataFrame)
_train_results = {} # Metrics, importances, etc.


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP: Generate data + train model
# ══════════════════════════════════════════════════════════════════════════════

def initialize():
    """Generate dataset and train (or load) the Random Forest model."""
    global _model, _df, _train_results

    os.makedirs("data", exist_ok=True)

    # Generate or load dataset
    if not os.path.exists(DATA_PATH):
        print("Generating synthetic dataset …")
        _df = generate_dataset(5000)
        _df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved: {_df.shape}")
    else:
        _df = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded: {_df.shape}")

    # Train or load model
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model …")
        _model.load(MODEL_PATH)
        # Restore train_results from saved artifacts
        _train_results = {
            "metrics": _model.metrics,
            "feature_importances": _model.feature_importances,
            "actual": _model.test_actual[:100],
            "predicted": _model.test_predicted[:100]
        }
    else:
        print("Training Random Forest model …")
        _train_results = _model.train(_df)
        _model.save(MODEL_PATH)
        print("Model trained and saved.")

    print(f"Model R² = {_model.metrics.get('r2', '?')}")


# ══════════════════════════════════════════════════════════════════════════════
# HTML ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/metadata")
def api_metadata():
    """Return dropdown options for the prediction form."""
    return jsonify({
        "districts": sorted(DISTRICTS),
        "crops": CROPS,
        "seasons": SEASONS
    })


@app.route("/api/metrics")
def api_metrics():
    """Return model evaluation metrics."""
    return jsonify(_model.metrics)


@app.route("/api/feature_importance")
def api_feature_importance():
    """Return feature importances sorted descending."""
    return jsonify(_model.feature_importances)


@app.route("/api/actual_vs_predicted")
def api_actual_vs_predicted():
    """Return first 100 test-set actual vs predicted pairs for charting."""
    data = list(zip(_model.test_actual[:100], _model.test_predicted[:100]))
    return jsonify({
        "actual":    [round(float(v), 2) for v, _ in data],
        "predicted": [round(float(v), 2) for _, v in data]
    })


@app.route("/api/district_summary")
def api_district_summary():
    """
    Return district-wise aggregated water demand metrics.
    Used for the dashboard heatmap/bar chart.
    """
    summary = _model.predict_district_summary(_df)
    return jsonify(summary)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept JSON body with feature values, return predicted water demand + recommendation.
    Example body:
    {
        "district": "Thanjavur",
        "crop_type": "Paddy (Samba)",
        "season": "Kharif (Jun-Oct)",
        "rainfall_mm": 1100,
        "temperature_c": 30,
        "reservoir_level": 0.75,
        "population_density": 680,
        "irrigated_area_ha": 8000,
        "groundwater_depth_m": 15,
        "soil_moisture_index": 0.65,
        "evapotranspiration_mm": 5.2,
        "industrial_usage_mcm": 2.5
    }
    """
    try:
        data = request.get_json(force=True)
        result = _model.predict(data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/optimization_recommendations")
def api_optimization_recommendations():
    """
    Generate district-level optimization recommendations based on
    model predictions and domain rules.
    """
    summary = _model.predict_district_summary(_df)

    recommendations = []
    for row in summary:
        dist = row["district"]
        demand = row["avg_predicted"]
        rainfall = row["avg_rainfall"]
        reservoir = row["avg_reservoir"]

        # Classify efficiency tier
        if demand < 300:
            tier = "Efficient"
            color = "#22c55e"
        elif demand < 600:
            tier = "Moderate"
            color = "#f59e0b"
        else:
            tier = "High Demand"
            color = "#ef4444"

        tips = []
        if rainfall < 700:
            tips.append("Micro-irrigation mandate")
        if reservoir < 0.45:
            tips.append("Groundwater recharge priority")
        if demand > 700:
            tips.append("Crop diversification to less water-intensive varieties")
        if not tips:
            tips.append("Maintain current usage patterns")

        recommendations.append({
            "district": dist,
            "avg_demand_mcm": round(demand, 1),
            "avg_rainfall_mm": round(rainfall, 1),
            "avg_reservoir": round(reservoir, 3),
            "efficiency_tier": tier,
            "tier_color": color,
            "recommendations": tips
        })

    # Sort by demand descending (highest need first)
    recommendations.sort(key=lambda x: x["avg_demand_mcm"], reverse=True)
    return jsonify(recommendations)


@app.route("/api/seasonal_trend")
def api_seasonal_trend():
    """Aggregate average demand by season for trend chart."""
    summary = (
        _df.groupby("season")["water_demand_mcm"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_demand", "std": "std_demand", "count": "n"})
        .round(2)
    )
    return jsonify(summary.to_dict(orient="records"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    initialize()
    app.run(debug=False, host="0.0.0.0", port=5000)
