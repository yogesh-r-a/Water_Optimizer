"""
model.py
Random Forest training, evaluation, and prediction logic for Tamil Nadu
Water Usage Optimizer.

WHY RANDOM FOREST for this problem?
─────────────────────────────────────────────────────────────────────────
1. Non-linear relationships: water demand has complex, non-linear
   interactions (rainfall × crop type × season). RF handles these natively
   without feature engineering.

2. Mixed feature types: our dataset has both numerical (rainfall, temp)
   and categorical (district, crop, season) features. RF works well with
   label-encoded categoricals.

3. Robustness to outliers: extreme drought/flood years create outliers;
   RF's bootstrap aggregation dampens their effect.

4. Feature importance: built-in Gini impurity-based importance helps
   domain experts understand which variables drive water demand.

5. No distributional assumptions: unlike linear regression, RF doesn't
   assume Gaussian residuals — critical for skewed agricultural data.

6. Good out-of-the-box performance: with limited domain knowledge for
   hyperparameter initialization, RF is forgiving and stable.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Feature columns used for training
FEATURE_COLS = [
    "district", "crop_type", "season",
    "rainfall_mm", "temperature_c", "reservoir_level",
    "population_density", "irrigated_area_ha",
    "groundwater_depth_m", "soil_moisture_index",
    "evapotranspiration_mm", "industrial_usage_mcm"
]
TARGET_COL = "water_demand_mcm"

# Categorical columns that need label encoding
CAT_COLS = ["district", "crop_type", "season"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


class WaterDemandModel:
    """
    Encapsulates the complete ML pipeline:
      1. Label encoding for categoricals
      2. StandardScaler for numericals (RF doesn't require scaling,
         but we include it for potential future model swaps)
      3. RandomForestRegressor with tuned hyperparameters
    """

    def __init__(self):
        self.label_encoders = {}   # One LabelEncoder per categorical col
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = FEATURE_COLS
        self.metrics = {}
        self.feature_importances = {}

    # ── Preprocessing ──────────────────────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns. If fit=True, fit the encoders."""
        df = df.copy()
        for col in CAT_COLS:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Full preprocessing: encode → select features → scale."""
        df = self._encode_categoricals(df, fit=fit)
        X = df[FEATURE_COLS].values.astype(float)
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train with RandomizedSearchCV hyperparameter tuning.

        Hyperparameter rationale:
        - n_estimators 200-400: enough trees to stabilize variance without
          excessive compute; RF error plateaus quickly after ~200 trees.
        - max_depth 10-30: prevents overfitting; shallow enough for generalization
          on 5000 samples.
        - min_samples_split 2-10: controls leaf purity vs. overfitting trade-off.
        - max_features 'sqrt'/'log2': the key RF randomization param;
          'sqrt' is standard for regression problems.
        - bootstrap True: bagging reduces variance (core RF advantage).
        """
        X = self.preprocess(df, fit=True)
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Hyperparameter search space
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.7],
            "bootstrap": [True],
        }

        base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

        # RandomizedSearchCV: faster than GridSearch; 20 iterations covers
        # the space well enough for 5000-sample datasets
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        y_pred = self.model.predict(X_test)

        # ── Evaluation metrics ─────────────────────────────────────────
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation R² for robustness estimate
        cv_r2 = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        ).mean()

        self.metrics = {
            "rmse": round(float(rmse), 3),
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 4),
            "cv_r2": round(float(cv_r2), 4),
            "best_params": search.best_params_,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        # ── Feature importances ────────────────────────────────────────
        importances = self.model.feature_importances_
        self.feature_importances = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(FEATURE_COLS, importances),
                key=lambda x: x[1], reverse=True
            )
        }

        # Store test predictions for actual vs predicted chart
        self.test_actual = y_test.tolist()
        self.test_predicted = y_pred.tolist()

        return {
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
            "actual": self.test_actual[:100],     # first 100 samples for chart
            "predicted": self.test_predicted[:100]
        }

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, input_data: dict) -> dict:
        """
        Predict water demand for a single input.
        input_data keys must match FEATURE_COLS.
        """
        df_in = pd.DataFrame([input_data])
        X = self.preprocess(df_in, fit=False)
        pred = self.model.predict(X)[0]

        # Optimization tip: compare against district average
        recommendation = self._get_recommendation(input_data, pred)

        return {
            "predicted_demand_mcm": round(float(pred), 2),
            "recommendation": recommendation
        }

    def _get_recommendation(self, inp: dict, pred: float) -> str:
        """Rule-augmented ML recommendation for actionable insight."""
        tips = []

        if inp.get("rainfall_mm", 0) < 400:
            tips.append("Low rainfall detected — consider drip/sprinkler irrigation to cut usage by 30-40%.")
        if inp.get("reservoir_level", 1) < 0.4:
            tips.append("Reservoir below 40% — prioritize domestic supply; defer non-critical irrigation.")
        if inp.get("crop_type", "") in ["Paddy (Kuruvai)", "Paddy (Samba)", "Sugarcane"]:
            tips.append("High-water crop selected — consider alternate wetting & drying (AWD) for paddy to save 25% water.")
        if inp.get("groundwater_depth_m", 0) > 40:
            tips.append("Deep groundwater (>40m) — groundwater extraction cost is high; rely on surface water or recycled water.")
        if inp.get("soil_moisture_index", 1) > 0.7:
            tips.append("High soil moisture — reduce irrigation scheduling frequency; risk of waterlogging.")
        if pred > 800:
            tips.append(f"Predicted demand ({pred:.0f} MCM) is high — explore precision agriculture and canal lining to reduce losses.")

        if not tips:
            tips.append("Water usage is within optimal range. Continue current practices.")

        return " | ".join(tips)

    # ── District-wise aggregation ──────────────────────────────────────────
    def predict_district_summary(self, df: pd.DataFrame) -> list:
        """Return per-district average predicted demand for dashboard chart."""
        X = self.preprocess(df.copy(), fit=False)
        preds = self.model.predict(X)
        df = df.copy()
        df["predicted"] = preds
        summary = (
            df.groupby("district")
            .agg(
                avg_predicted=("predicted", "mean"),
                avg_actual=("water_demand_mcm", "mean"),
                avg_rainfall=("rainfall_mm", "mean"),
                avg_reservoir=("reservoir_level", "mean")
            )
            .reset_index()
            .round(2)
        )
        return summary.to_dict(orient="records")

    # ── Serialization ──────────────────────────────────────────────────────
    def save(self, path: str = "model_artifacts.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_encoders": self.label_encoders,
                "scaler": self.scaler,
                "metrics": self.metrics,
                "feature_importances": self.feature_importances,
                "test_actual": getattr(self, "test_actual", []),
                "test_predicted": getattr(self, "test_predicted", [])
            }, f)

    def load(self, path: str = "model_artifacts.pkl"):
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        self.model = artifacts["model"]
        self.label_encoders = artifacts["label_encoders"]
        self.scaler = artifacts["scaler"]
        self.metrics = artifacts["metrics"]
        self.feature_importances = artifacts["feature_importances"]
        self.test_actual = artifacts.get("test_actual", [])
        self.test_predicted = artifacts.get("test_predicted", [])
