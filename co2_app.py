import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# Load Model
# -------------------------------

model = joblib.load("co2_model.pkl")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("co2_emissions.csv")



# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="CO2 Emission Prediction",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📋 Project Information")
st.sidebar.markdown("### 🚗 Project")
st.sidebar.info("CO2 Emission Prediction")

st.sidebar.markdown("### 👩 Developed By")
st.sidebar.success("Group-5")

st.sidebar.markdown("### 🤖 Machine Learning")
st.sidebar.info("Regression")

st.sidebar.markdown("### 🧠 Algorithm")
st.sidebar.success("Random Forest Regressor")

st.sidebar.markdown("### ✅ Status")
st.sidebar.success("Ready")

# -------------------------------
# Main Title
# -------------------------------
st.title("🚗 CO2 Emission Prediction Dashboard")
st.markdown("---")

# -------------------------------
# Dashboard
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Target", "CO₂")

with col3:
    st.metric("Status", "Ready")

st.markdown("---")

# -------------------------------
# Inputs
# -------------------------------
st.header("🚘 Vehicle Information")

make = st.selectbox(
    "Make",
    sorted(df["make"].astype(str).unique())
)

model_name = st.selectbox(
    "Model",
    sorted(df["model"].astype(str).unique())
)

vehicle_class = st.selectbox(
    "Vehicle Class",
    sorted(df["vehicle_class"].astype(str).unique())
)

engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    value=2.0
)

cylinders = st.number_input(
    "Cylinders",
    min_value=1,
    value=4
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].astype(str).unique())
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].astype(str).unique())
)

fuel_city = st.number_input(
    "Fuel Consumption City (L/100km)",
    min_value=0.0,
    value=8.0
)

fuel_hwy = st.number_input(
    "Fuel Consumption Highway (L/100km)",
    min_value=0.0,
    value=6.0
)

fuel_comb = st.number_input(
    "Fuel Consumption Combined (L/100km)",
    min_value=0.0,
    value=7.0
)

fuel_mpg = st.number_input(
    "Fuel Consumption Combined (MPG)",
    min_value=0.0,
    value=35.0
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict CO₂ Emission"):

    try:

        # Create input dataframe
        input_data = pd.DataFrame({
            "make": [make],
            "model": [model_name],
            "vehicle_class": [vehicle_class],
            "engine_size": [engine_size],
            "cylinders": [cylinders],
            "transmission": [transmission],
            "fuel_type": [fuel_type],
            "fuel_consumption_city": [fuel_city],
            "fuel_consumption_hwy": [fuel_hwy],
            "fuel_consumption_comb(l/100km)": [fuel_comb],
            "fuel_consumption_comb(mpg)": [fuel_mpg]
        })

        # Feature Engineering
        input_data["Fuel_Efficiency"] = 100 / input_data["fuel_consumption_comb(l/100km)"]
        input_data["Engine_per_Cylinder"] = input_data["engine_size"] / input_data["cylinders"]
        input_data["Average_Fuel"] = (
            input_data["fuel_consumption_city"] +
            input_data["fuel_consumption_hwy"] +
            input_data["fuel_consumption_comb(l/100km)"]
        ) / 3

        # One-hot encoding
        input_data = pd.get_dummies(
            input_data,
            columns=[
                "make",
                "model",
                "vehicle_class",
                "transmission",
                "fuel_type"
            ]
        )

        # Match training columns
        expected_columns = model.feature_names_in_
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        prediction = model.predict(input_data)

        st.success(f"✅ Predicted CO₂ Emission: {prediction[0]:.2f} g/km")
        st.balloons()

    except Exception as e:
        st.error("Prediction Failed!")
        st.error(str(e))
