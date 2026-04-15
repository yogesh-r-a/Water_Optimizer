"""
data_generator.py
Generates a realistic synthetic dataset for Tamil Nadu water usage optimization.
Tamil Nadu has 38 districts with diverse agricultural zones, varying rainfall patterns
(northeast/southwest monsoons), and a mix of crops suited to its agro-climatic zones.
"""

import numpy as np
import pandas as pd

# Seed for reproducibility
np.random.seed(42)

# ── Tamil Nadu's 38 districts ──────────────────────────────────────────────
DISTRICTS = [
    "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
    "Tirunelveli", "Vellore", "Erode", "Thoothukudi", "Dindigul",
    "Thanjavur", "Tiruvannamalai", "Kanchipuram", "Cuddalore", "Nagapattinam",
    "Karur", "Namakkal", "Dharmapuri", "Krishnagiri", "Perambalur",
    "Ariyalur", "Villupuram", "Kallakurichi", "Ranipet", "Tirupathur",
    "Chengalpattu", "Tirupur", "Nilgiris", "Ramanathapuram", "Sivaganga",
    "Virudhunagar", "Pudukkottai", "Thiruvarur", "Mayiladuthurai",
    "Tenkasi", "Tirupattur", "Kanniyakumari", "Theni"
]

# Crop types common in Tamil Nadu, grouped by water intensity
CROPS = [
    "Paddy (Kuruvai)", "Paddy (Samba)", "Sugarcane", "Banana",
    "Cotton", "Groundnut", "Maize", "Turmeric",
    "Coconut", "Mango", "Tomato", "Onion",
    "Tapioca", "Pulses", "Millets", "Vegetables"
]

SEASONS = ["Kharif (Jun-Oct)", "Rabi (Nov-Mar)", "Zaid (Apr-May)", "Year-round"]

# Water demand base values (MCM - Million Cubic Meters) per crop per season
# Higher for paddy/sugarcane, lower for millets/pulses
CROP_WATER_BASE = {
    "Paddy (Kuruvai)": 1800, "Paddy (Samba)": 2200, "Sugarcane": 2500,
    "Banana": 1600, "Cotton": 900, "Groundnut": 700, "Maize": 750,
    "Turmeric": 1100, "Coconut": 1200, "Mango": 800, "Tomato": 850,
    "Onion": 650, "Tapioca": 950, "Pulses": 500, "Millets": 450,
    "Vegetables": 800
}

# District-level base population density (persons per sq km, approximate)
DISTRICT_DENSITY = {
    "Chennai": 26553, "Coimbatore": 752, "Madurai": 1052, "Tiruchirappalli": 781,
    "Salem": 645, "Tirunelveli": 479, "Vellore": 787, "Erode": 556,
    "Thoothukudi": 484, "Dindigul": 387, "Thanjavur": 680, "Tiruvannamalai": 545,
    "Kanchipuram": 1069, "Cuddalore": 638, "Nagapattinam": 543,
    "Karur": 424, "Namakkal": 567, "Dharmapuri": 337, "Krishnagiri": 357,
    "Perambalur": 253, "Ariyalur": 278, "Villupuram": 424,
    "Kallakurichi": 368, "Ranipet": 720, "Tirupattur": 481,
    "Chengalpattu": 1102, "Tirupur": 896, "Nilgiris": 220,
    "Ramanathapuram": 222, "Sivaganga": 366, "Virudhunagar": 515,
    "Pudukkottai": 315, "Thiruvarur": 475, "Mayiladuthurai": 617,
    "Tenkasi": 412, "Tirupattur": 481, "Kanniyakumari": 1169, "Theni": 397
}

# Average annual rainfall (mm) by district — northeast/southwest monsoon split
DISTRICT_RAINFALL = {
    "Chennai": 1400, "Coimbatore": 700, "Madurai": 850, "Tiruchirappalli": 830,
    "Salem": 900, "Tirunelveli": 650, "Vellore": 1050, "Erode": 800,
    "Thoothukudi": 600, "Dindigul": 800, "Thanjavur": 1100, "Tiruvannamalai": 1000,
    "Kanchipuram": 1300, "Cuddalore": 1250, "Nagapattinam": 1300,
    "Karur": 750, "Namakkal": 850, "Dharmapuri": 950, "Krishnagiri": 1100,
    "Perambalur": 780, "Ariyalur": 900, "Villupuram": 1100,
    "Kallakurichi": 1000, "Ranipet": 1100, "Tirupattur": 950,
    "Chengalpattu": 1350, "Tirupur": 750, "Nilgiris": 2500,
    "Ramanathapuram": 750, "Sivaganga": 800, "Virudhunagar": 700,
    "Pudukkottai": 900, "Thiruvarur": 1100, "Mayiladuthurai": 1200,
    "Tenkasi": 1200, "Kanniyakumari": 1600, "Theni": 900
}

# Reservoir storage capacity index (0-1 scale, normalized)
DISTRICT_RESERVOIR = {d: np.random.uniform(0.3, 0.9) for d in DISTRICTS}
# Delta districts with major river systems get higher reservoir capacity
for d in ["Thanjavur", "Thiruvarur", "Nagapattinam", "Mayiladuthurai", "Tiruchirappalli"]:
    DISTRICT_RESERVOIR[d] = np.random.uniform(0.7, 0.95)


def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic water usage dataset for Tamil Nadu.

    Feature engineering rationale:
    - Agricultural demand: dominant factor (~70% of TN water use)
    - Rainfall: negatively correlated with irrigation need
    - Reservoir level: affects water availability (supply-side)
    - Temperature: proxy for evapotranspiration (ET0)
    - Population density: drives domestic + industrial demand
    - Season: captures monsoon/non-monsoon dynamics
    """
    records = []

    for _ in range(n_samples):
        district = np.random.choice(DISTRICTS)
        crop = np.random.choice(CROPS)
        season = np.random.choice(SEASONS)

        # ── Rainfall: base + seasonal noise ──────────────────────────────
        base_rainfall = DISTRICT_RAINFALL.get(district, 900)
        season_factor = {
            "Kharif (Jun-Oct)": 1.4,   # SW monsoon — high rainfall
            "Rabi (Nov-Mar)": 0.7,     # NE monsoon — moderate
            "Zaid (Apr-May)": 0.3,     # Summer — very dry
            "Year-round": 1.0
        }[season]
        rainfall_mm = max(0, base_rainfall * season_factor + np.random.normal(0, 80))

        # ── Reservoir level (0-1): influenced by rainfall ─────────────────
        reservoir_base = DISTRICT_RESERVOIR.get(district, 0.5)
        reservoir_level = np.clip(
            reservoir_base + (rainfall_mm / base_rainfall - 1) * 0.2 + np.random.normal(0, 0.05),
            0.05, 1.0
        )

        # ── Temperature (°C): seasonal variation ─────────────────────────
        temp_base = {"Kharif (Jun-Oct)": 32, "Rabi (Nov-Mar)": 26,
                     "Zaid (Apr-May)": 38, "Year-round": 30}[season]
        temperature_c = temp_base + np.random.normal(0, 2)

        # ── Population density ────────────────────────────────────────────
        pop_density = DISTRICT_DENSITY.get(district, 500) + np.random.normal(0, 30)

        # ── Irrigated area (hectares): crop-specific, with noise ──────────
        crop_area_ha = np.random.uniform(500, 15000)

        # ── Groundwater depth (m): deeper = harder to extract ────────────
        groundwater_depth_m = np.random.uniform(5, 60) + (40 - rainfall_mm / 50)
        groundwater_depth_m = np.clip(groundwater_depth_m, 3, 80)

        # ── Soil moisture index (0-1) ─────────────────────────────────────
        soil_moisture = np.clip(
            0.3 + (rainfall_mm / 3000) + np.random.normal(0, 0.05), 0.1, 0.95
        )

        # ── Evapotranspiration (mm/day): Penman-Monteith proxy ───────────
        et0 = 0.5 * temperature_c - 0.003 * rainfall_mm / 30 + np.random.normal(0, 0.5)
        et0 = np.clip(et0, 2, 12)

        # ── Industrial water use (MCM): correlated with urban density ─────
        industrial_mcm = pop_density * 0.0003 + np.random.uniform(0, 5)

        # ── TARGET: Total water demand (MCM) ─────────────────────────────
        # Agricultural component: crop base * area * season drought factor
        agri_base = CROP_WATER_BASE[crop]
        drought_multiplier = max(0.6, 1.5 - rainfall_mm / base_rainfall)
        agri_demand = (agri_base * (crop_area_ha / 10000) * drought_multiplier
                       * (1 + et0 * 0.02))

        # Domestic demand: population-driven
        domestic_demand = pop_density * 0.00015 * 150  # 150 lpcd

        # Reservoir reduces demand by increasing supply availability
        supply_efficiency = 1 - reservoir_level * 0.25

        # Total water demand with noise
        total_demand_mcm = (
            (agri_demand + domestic_demand + industrial_mcm) * supply_efficiency
            + np.random.normal(0, 15)
        )
        total_demand_mcm = max(10, total_demand_mcm)

        records.append({
            "district": district,
            "crop_type": crop,
            "season": season,
            "rainfall_mm": round(rainfall_mm, 1),
            "temperature_c": round(temperature_c, 1),
            "reservoir_level": round(reservoir_level, 3),
            "population_density": round(pop_density, 0),
            "irrigated_area_ha": round(crop_area_ha, 0),
            "groundwater_depth_m": round(groundwater_depth_m, 1),
            "soil_moisture_index": round(soil_moisture, 3),
            "evapotranspiration_mm": round(et0, 2),
            "industrial_usage_mcm": round(industrial_mcm, 2),
            "water_demand_mcm": round(total_demand_mcm, 2)
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("data/tn_water_usage.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df.describe())
