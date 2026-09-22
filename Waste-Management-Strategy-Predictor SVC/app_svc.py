import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Waste Management Strategy Predictor",
    page_icon="♻️",
    layout="centered",
)

MODEL_PATH = "waste management_model.pkl"
FEATURE_NAMES = [
    "waste_type",
    "material_composition",
    "recycling_potential",
    "toxicity_level",
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

st.title("♻️ Waste Management Strategy Predictor")
st.write(
    "This app uses a pre-trained Support Vector Classifier (SVC) to predict "
    "whether a waste stream should be managed as **Recyclable** or "
    "**Non-Recyclable**, based on four key indicators."
)

if model is None:
    st.error(
        f"Could not find `{MODEL_PATH}`. Place the trained model file in the "
        "same folder as this app (the file produced by your notebook via "
        "`joblib.dump(model, 'waste management_model.pkl')`)."
    )
    st.stop()

st.divider()

# ----------------------------------------------------------------------------
# Input mode: single prediction vs. batch (CSV) prediction
# ----------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔎 Single Prediction", "📄 Batch Prediction (CSV)"])

# ---------------------------- Single prediction -----------------------------
with tab_single:
    st.subheader("Enter Waste Metrics")

    col1, col2 = st.columns(2)
    with col1:
        waste_type = st.number_input(
            "Waste Type (category code)", min_value=0, max_value=10, value=1, step=1
        )
        material_composition = st.slider(
            "Material Composition (0.0 - 1.0)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        )
    with col2:
        recycling_potential = st.slider(
            "Recycling Potential (0.0 - 1.0)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        )
        toxicity_level = st.number_input(
            "Toxicity Level", min_value=0.0, value=50.0, step=1.0, format="%.2f"
        )

    if st.button("Predict Management Strategy", type="primary"):
        input_df = pd.DataFrame(
            [[waste_type, material_composition, recycling_potential, toxicity_level]],
            columns=FEATURE_NAMES,
        )

        prediction = model.predict(input_df)[0]
        label_map = {0: "Non-Recyclable", 1: "Recyclable"}
        result_label = label_map.get(prediction, str(prediction))

        st.markdown("### Result")
        if prediction == 1:
            st.success(f"✅ Prediction: **{result_label}**")
        else:
            st.warning(f"⚠️ Prediction: **{result_label}**")

        # SVC with default settings has no predict_proba unless probability=True
        # was set during training, so we show the decision function instead.
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame(
                    {"Class": ["Non-Recyclable", "Recyclable"], "Probability": proba}
                )
                st.bar_chart(proba_df.set_index("Class"))
            except Exception:
                pass

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

                result_df = batch_df.copy()
                result_df["prediction"] = np.where(
                    preds == 1, "Recyclable", "Non-Recyclable"
                )

                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_batch)[:, 1]
                        result_df["recyclable_probability"] = probs
                    except Exception:
                        pass

                st.success(f"Predicted {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv_out,
                    file_name="waste_management_predictions.csv",
                    mime="text/csv",
                )

st.divider()
with st.expander("ℹ️ About the features"):
    st.markdown(
        """
- **waste_type**: Encoded category of waste type
- **material_composition**: Normalized measure of material composition (0 to 1)
- **recycling_potential**: Normalized recycling potential score (0 to 1)
- **toxicity_level**: Toxicity level measurement

The model is a **Support Vector Classifier** (RBF kernel) trained offline; this app
only loads the saved model (`waste management_model.pkl`) and serves predictions —
no training happens here.
"""
    )
