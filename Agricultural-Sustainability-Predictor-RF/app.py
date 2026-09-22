import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agricultural Sustainability Predictor",
    page_icon="🌱",
    layout="centered",
)

MODEL_PATH = "agricultural_sustainability_model.pkl"
FEATURE_NAMES = [
    "soil_health",
    "crop_yield",
    "water_usage",
    "carbon_footprint",
    "fertilizer_use",
]


# ----------------------------------------------------------------------------
# Load the already-trained model (no training happens in this file)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


model = load_model(MODEL_PATH)

st.title("🌱 Agricultural Sustainability Predictor")
st.write(
    "This app uses a pre-trained Random Forest model to predict whether a "
    "farm's practices are **Sustainable** or **Unsustainable**, based on five "
    "key indicators."
)

if model is None:
    st.error(
        f"Could not find `{MODEL_PATH}`. Place the trained model file in the "
        "same folder as this app (the file produced by your notebook via "
        "`joblib.dump(model, 'agricultural_sustainability_model.pkl')`)."
    )
    st.stop()

st.divider()

# ----------------------------------------------------------------------------
# Input mode: single prediction vs. batch (CSV) prediction
# ----------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔎 Single Prediction", "📄 Batch Prediction (CSV)"])

# ---------------------------- Single prediction -----------------------------
with tab_single:
    st.subheader("Enter Farm Metrics")

    col1, col2 = st.columns(2)
    with col1:
        soil_health = st.slider(
            "Soil Health (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        crop_yield = st.number_input(
            "Crop Yield", min_value=0.0, value=5000.0, step=100.0, format="%.2f"
        )
        water_usage = st.number_input(
            "Water Usage", min_value=0.0, value=2500.0, step=100.0, format="%.2f"
        )
    with col2:
        carbon_footprint = st.number_input(
            "Carbon Footprint", min_value=0.0, value=120.0, step=10.0, format="%.2f"
        )
        fertilizer_use = st.number_input(
            "Fertilizer Use", min_value=0.0, value=150.0, step=10.0, format="%.2f"
        )

    if st.button("Predict Sustainability", type="primary"):
        input_df = pd.DataFrame(
            [[soil_health, crop_yield, water_usage, carbon_footprint, fertilizer_use]],
            columns=FEATURE_NAMES,
        )

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label_map = {0: "Unsustainable", 1: "Sustainable"}
        result_label = label_map.get(prediction, str(prediction))

        st.markdown("### Result")
        if prediction == 1:
            st.success(f"✅ Prediction: **{result_label}**")
        else:
            st.warning(f"⚠️ Prediction: **{result_label}**")

        proba_df = pd.DataFrame(
            {"Class": ["Unsustainable", "Sustainable"], "Probability": proba}
        )
        st.bar_chart(proba_df.set_index("Class"))

        with st.expander("Show input data"):
            st.dataframe(input_df, use_container_width=True)

# ---------------------------- Batch prediction -------------------------------
with tab_batch:
    st.subheader("Upload a CSV for Batch Prediction")
    st.caption(
        "The CSV must contain these columns: "
        + ", ".join(f"`{c}`" for c in FEATURE_NAMES)
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            batch_df = None

        if batch_df is not None:
            missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required column(s): {', '.join(missing_cols)}")
            else:
                X_batch = batch_df[FEATURE_NAMES]
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]

                result_df = batch_df.copy()
                result_df["prediction"] = np.where(preds == 1, "Sustainable", "Unsustainable")
                result_df["sustainable_probability"] = probs

                st.success(f"Predicted {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv_out,
                    file_name="sustainability_predictions.csv",
                    mime="text/csv",
                )

st.divider()
with st.expander("ℹ️ About the features"):
    st.markdown(
        """
- **soil_health**: Normalized soil health score (0 to 1)
- **crop_yield**: Crop yield amount
- **water_usage**: Water usage amount
- **carbon_footprint**: Carbon footprint measure
- **fertilizer_use**: Fertilizer use amount

The model is a **Random Forest Classifier** (100 trees) trained offline; this app only
loads the saved model (`agricultural_sustainability_model.pkl`) and serves predictions —
no training happens here.
"""
    )
