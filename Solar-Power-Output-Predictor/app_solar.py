import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Solar Power Output Predictor", layout="centered")

MODEL_PATH = "solar_power_prediction_model.pkl"

st.title("☀️ Solar Power Output Predictor")
st.write(
    "Predicts **solar power output (watts)** from temperature, humidity, "
    "solar irradiance, and wind speed, using a pre-trained Linear Regression model."
)

# ---------------------------------------------------------
# Load the pre-trained model
# ---------------------------------------------------------
st.sidebar.header("Model")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"])


@st.cache_resource
def load_model_from_path(path):
    return joblib.load(path)


model = None
if uploaded_model is not None:
    model = joblib.load(uploaded_model)
else:
    try:
        model = load_model_from_path(MODEL_PATH)
        st.sidebar.success(f"Loaded `{MODEL_PATH}` found alongside the app.")
    except FileNotFoundError:
        st.error(
            f"Could not find `{MODEL_PATH}` next to app.py. "
            "Please place the trained model file in the same folder as this app, "
            "or upload it using the sidebar."
        )
        st.stop()

# ---------------------------------------------------------
# Input UI
# ---------------------------------------------------------
st.subheader("Enter environmental conditions")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", 10.0, 35.0, 22.0, 0.1)
    humidity = st.slider("Humidity (%)", 20.0, 100.0, 60.0, 0.1)
with col2:
    solar_irradiance = st.slider("Solar Irradiance (W/m²)", 100.0, 1000.0, 500.0, 1.0)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 10.0, 5.0, 0.1)

st.markdown("---")

if st.button("Predict Solar Power Output", type="primary"):
    new_data = np.array([[temperature, humidity, solar_irradiance, wind_speed]])
    predicted_output = model.predict(new_data)[0]
    st.success(f"Predicted Solar Power Output: **{predicted_output:.2f} watts**")

st.caption(
    "Inputs: temperature, humidity, solar_irradiance, wind_speed — "
    "in the same order the model was trained on."
)
