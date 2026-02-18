import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =================================================
# Load Models (Portable Paths)
# =================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

yield_model = joblib.load(MODEL_DIR / "CRA_Final_Yield_Prediction_Model.joblib")
resilience_model = joblib.load(MODEL_DIR / "CRA_Final_Resilient_Classifier_Model.joblib")

# =================================================
# Helper: Align input to one-hot trained model
# =================================================
def align_to_trained_features(input_df, model):
    """
    Align raw input_df to the exact feature space
    used during training of a non-pipeline model
    (one-hot encoded).
    """
    trained_features = model.feature_names_in_
    aligned = pd.DataFrame(0, index=input_df.index, columns=trained_features)

    for col in trained_features:
        # Numerical features
        if col in input_df.columns:
            aligned[col] = input_df[col].values
        else:
            # Handle one-hot categorical features
            for raw_col in input_df.columns:
                val = input_df[raw_col].iloc[0]
                one_hot_col = f"{raw_col}_{val}"
                if one_hot_col == col:
                    aligned[col] = 1

    return aligned

# =================================================
# Streamlit Config
# =================================================
st.set_page_config(
    page_title="Climate Resilient Agriculture",
    layout="wide"
)

st.title("🌱 Climate Resilient Agriculture – Decision Support System")

st.markdown("""
This system provides:
- 🌾 **Climate-Aware Yield Prediction**
- 🛡️ **Climate Resilience Classification**

Models are trained using **climate, soil, air quality, and water variables**
to support **sustainable agricultural decision-making**.
""")

# =================================================
# Sidebar Inputs
# =================================================
st.sidebar.header("🌦️ Input Parameters")

def get_user_input():
    return pd.DataFrame([{
        "Year": st.sidebar.slider("Year", 2000, 2030, 2020),
        "Avg_Temperature": st.sidebar.slider("Avg Temperature (°C)", 15.0, 35.0, 25.0),
        "Temp_Anomaly": st.sidebar.slider("Temperature Anomaly (°C)", -5.0, 5.0, 0.0),
        "Rainfall_mm": st.sidebar.slider("Rainfall (mm)", 300, 1500, 800),
        "Relative_Humidity": st.sidebar.slider("Relative Humidity (%)", 30, 100, 65),
        "Heatwave_Days": st.sidebar.slider("Heatwave Days", 0, 20, 2),
        "Dry_Spell_Count": st.sidebar.slider("Dry Spell Count", 0, 20, 5),
        "Soil_Organic_Carbon": st.sidebar.slider("Soil Organic Carbon", 0.4, 0.8, 0.55),
        "Soil_pH": st.sidebar.slider("Soil pH", 5.5, 8.5, 6.8),
        "Water_Holding_Capacity": st.sidebar.slider("Water Holding Capacity (%)", 30, 60, 40),
        "Electrical_Conductivity": st.sidebar.slider("Electrical Conductivity", 0.5, 2.0, 1.1),
        "Irrigation_Coverage": st.sidebar.slider("Irrigation Coverage (%)", 0, 60, 30),
        "Groundwater_Depth": st.sidebar.slider("Groundwater Depth (m)", 10, 50, 30),
        "Wind_Speed": st.sidebar.slider("Wind Speed (m/s)", 0.5, 6.0, 2.5),
        "Solar_Radiation": st.sidebar.slider("Solar Radiation", 10.0, 30.0, 20.0),
        "CO2_Concentration": st.sidebar.slider("CO₂ Concentration (ppm)", 380, 480, 420),
        "PM2_5": st.sidebar.slider("PM2.5", 5, 300, 80),
        "PM10": st.sidebar.slider("PM10", 10, 500, 150),
        "Aerosol_Optical_Depth": st.sidebar.slider("Aerosol Optical Depth", 0.05, 3.0, 1.2),
        "Ground_Level_Ozone": st.sidebar.slider("Ground-Level Ozone (ppb)", 5.0, 120.0, 40.0),
        "Season": st.sidebar.selectbox("Season", ["Kharif", "Rabi", "Summer", "Autumn", "Winter", "Whole year"]),
        "State": st.sidebar.selectbox("State", ["Maharashtra", "Punjab", "Tamil Nadu", "Andhra Pradesh", "Karnataka"]),
        "District": st.sidebar.selectbox("District", ["Pune", "Nagpur", "Chennai", "Amritsar", "Bengaluru"]),
        "Seed_Variety": st.sidebar.selectbox("Seed Variety", ["Local", "Hybrid", "HYV", "Traditional"]),
        "Irrigation_Source": st.sidebar.selectbox("Irrigation Source", ["Canal", "Rainfed", "Other"])
    }])

input_df = get_user_input()

# =================================================
# Climate Stress Index (used in resilience model)
# =================================================
stress_features = ["Heatwave_Days", "Dry_Spell_Count", "Temp_Anomaly"]

stress_mean = np.array([2.5, 5.0, 0.0])
stress_std  = np.array([2.0, 4.0, 2.0])

stress_vals = input_df[stress_features].values.astype(float)

input_df["Climate_Stress_Index"] = (
    (stress_vals - stress_mean) / stress_std
).mean(axis=1)

# =================================================
# Predictions
# =================================================
# Yield model → pipeline → raw input OK
yield_prediction = yield_model.predict(input_df)[0]

# Resilience model → one-hot encoded input required
resilience_input = align_to_trained_features(input_df, resilience_model)
resilience_prediction = resilience_model.predict(resilience_input)[0]

# =================================================
# Output
# =================================================
st.subheader("📊 Predictions")

col1, col2 = st.columns(2)

with col1:
    st.metric("🌾 Predicted Yield (kg/ha)", f"{yield_prediction:.0f}")

with col2:
    st.metric("🛡️ Climate Resilience Level", resilience_prediction)

st.subheader("🧠 Interpretation")

if resilience_prediction == "High":
    st.success("High resilience: system can sustain yield under climate stress.")
elif resilience_prediction == "Medium":
    st.warning("Moderate resilience: adaptive measures are recommended.")
else:
    st.error("Low resilience: high vulnerability to climate extremes.")
