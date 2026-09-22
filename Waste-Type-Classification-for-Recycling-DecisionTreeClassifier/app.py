import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Waste Type Classification",
    page_icon="♻️",
    layout="wide",
)

FEATURES = ["Weight", "Moisture", "Hardness", "Magnetic", "Biodegradable"]
TARGET = "WasteType"
WASTE_CATEGORIES = ["Plastic", "Metal", "Paper", "Glass", "Organic"]

FEATURE_INFO = {
    "Weight": (1.0, 500.0, "Weight of the item in grams."),
    "Moisture": (0.0, 100.0, "Moisture content, as a percentage."),
    "Hardness": (1.0, 10.0, "Hardness score (higher = harder material)."),
    "Magnetic": (0, 1, "1 if the item is magnetic (e.g. metal), else 0."),
    "Biodegradable": (0, 1, "1 if the item is biodegradable (e.g. organic/paper), else 0."),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data(num_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Fallback demo dataset with the same schema as 'waste_data.csv'
    (Weight, Moisture, Hardness, Magnetic, Biodegradable -> WasteType).
    The original CSV wasn't provided with the lab document, so this generates a
    plausible synthetic stand-in with rough real-world tendencies per category.
    Replace with the real CSV for accurate results.
    """
    rng = np.random.RandomState(seed)
    profiles = {
        "Plastic":       dict(weight=(5, 60),   moisture=(0, 10),  hardness=(2, 5), magnetic=0, bio=0),
        "Metal":         dict(weight=(50, 400), moisture=(0, 5),   hardness=(7, 10), magnetic=1, bio=0),
        "Paper":         dict(weight=(2, 40),   moisture=(10, 40), hardness=(1, 3), magnetic=0, bio=1),
        "Glass":         dict(weight=(80, 500), moisture=(0, 5),   hardness=(6, 9), magnetic=0, bio=0),
        "Organic":       dict(weight=(20, 300), moisture=(50, 95), hardness=(1, 3), magnetic=0, bio=1),
    }
    rows = []
    per_class = num_samples // len(profiles)
    for label, p in profiles.items():
        for _ in range(per_class):
            weight = rng.uniform(*p["weight"])
            moisture = rng.uniform(*p["moisture"])
            hardness = rng.uniform(*p["hardness"])
            magnetic = p["magnetic"] if rng.rand() > 0.05 else 1 - p["magnetic"]
            bio = p["bio"] if rng.rand() > 0.05 else 1 - p["bio"]
            rows.append([weight, moisture, hardness, magnetic, bio, label])

    df = pd.DataFrame(rows, columns=FEATURES + [TARGET])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_resource
def train_models(df: pd.DataFrame, test_size: float, random_state: int):
    X = df[FEATURES]
    y_raw = df[TARGET]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    return encoder, X_train, X_test, y_train, y_test, dt, dt_pred, lr, lr_pred


# ----------------------------------------------------------------------------
# Sidebar — data source & model controls
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Data & Model Settings")

data_source = st.sidebar.radio(
    "Dataset source",
    ["Upload waste_data.csv", "Use built-in demo data"],
)

if data_source == "Upload waste_data.csv":
    uploaded = st.sidebar.file_uploader(
        "CSV with columns: " + ", ".join(FEATURES + [TARGET]), type=["csv"]
    )
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing_cols = set(FEATURES + [TARGET]) - set(data.columns)
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {missing_cols}")
            st.stop()
        data = data.dropna()
        st.sidebar.success(f"Loaded {len(data)} rows.")
    else:
        st.sidebar.info("Upload your CSV, or switch to the demo dataset below.")
        data = generate_demo_data()
        st.sidebar.caption("Currently showing demo data until a CSV is uploaded.")
else:
    num_samples = st.sidebar.slider("Number of demo samples", 50, 1000, 250, step=50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    data = generate_demo_data(num_samples=num_samples, seed=seed)
    st.sidebar.warning(
        "This is synthetic demo data (the original waste_data.csv wasn't provided). "
        "Upload the real file for accurate results."
    )

test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Train/test split random state", value=42, step=1)

encoder, X_train, X_test, y_train, y_test, dt, dt_pred, lr, lr_pred = train_models(
    data, test_size, random_state
)
class_names = list(encoder.classes_)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Reproduces the lab notebook: load data → encode target → train/test split → "
    "Decision Tree + Logistic Regression → evaluate & compare, plus an interactive predictor."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("♻️ Waste Type Classification for Recycling")
st.write(
    "Compare a **Decision Tree** and a **Logistic Regression** model that classify "
    "waste items into categories (e.g. Plastic, Metal, Paper, Glass, Organic) based on "
    "their physical characteristics."
)

tab_predict, tab_explore, tab_performance, tab_compare, tab_about = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "📈 Model Performance", "⚖️ Compare Models", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1: Interactive prediction
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Try it yourself")
    st.write("Describe a waste item, then see what each model predicts.")

    model_choice = st.radio("Model to use", ["Decision Tree", "Logistic Regression"], horizontal=True)

    col1, col2 = st.columns([1, 1])

    input_values = {}
    with col1:
        for feat in FEATURES:
            lo, hi, help_text = FEATURE_INFO[feat]
            if feat in ("Magnetic", "Biodegradable"):
                input_values[feat] = st.selectbox(feat, [0, 1], help=help_text)
            else:
                default = float(data[feat].mean())
                step = (hi - lo) / 100
                input_values[feat] = st.slider(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=float(np.clip(default, lo, hi)), step=float(step), help=help_text,
                )

    input_df = pd.DataFrame([input_values])[FEATURES]
    model = dt if model_choice == "Decision Tree" else lr
    pred_encoded = model.predict(input_df)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(input_df)[0]

    with col2:
        st.markdown("#### Prediction")
        st.success(f"🏷️ **{pred_label}**")

        proba_df = pd.DataFrame({"WasteType": class_names, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.dataframe(proba_df.set_index("WasteType"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(proba_df["WasteType"], proba_df["Probability"], color="#5cb85c")
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Batch prediction")
    st.write("Upload a CSV with the same feature columns to classify many items at once.")
    batch_file = st.file_uploader(
        "CSV with columns: " + ", ".join(FEATURES), type=["csv"], key="batch_upload"
    )
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = set(FEATURES) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            batch_model = dt if model_choice == "Decision Tree" else lr
            preds = batch_model.predict(batch_df[FEATURES])
            batch_df["predicted_WasteType"] = encoder.inverse_transform(preds)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "⬇️ Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="waste_predictions.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Tab 2: Explore the data
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"Full dataset: {data.shape[0]} rows × {data.shape[1]} columns")

    st.subheader("Class balance")
    counts = data[TARGET].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    counts.plot(kind="bar", color="#5cb85c", ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("")
    st.pyplot(fig1)

    st.subheader("Feature distributions by waste type")
    feat_choice = st.selectbox("Choose a feature", FEATURES)
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    sns.boxplot(data=data, x=TARGET, y=feat_choice, ax=ax2)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.subheader("Correlation heatmap (numeric features)")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(data[FEATURES].corr(), annot=True, cmap="Blues", ax=ax3)
    st.pyplot(fig3)

# ----------------------------------------------------------------------------
# Tab 3: Model performance
# ----------------------------------------------------------------------------
with tab_performance:
    model_tab_choice = st.radio(
        "Show performance for", ["Decision Tree", "Logistic Regression"], horizontal=True, key="perf_choice"
    )
    if model_tab_choice == "Decision Tree":
        pred = dt_pred
        clf = dt
    else:
        pred = lr_pred
        clf = lr

    acc = accuracy_score(y_test, pred)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Train samples", len(X_train))
    m3.metric("Test samples", len(X_test))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, pred)
        fig4, ax4 = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax4,
        )
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        plt.xticks(rotation=30)
        plt.yticks(rotation=0)
        st.pyplot(fig4)

    with col_b:
        st.markdown("#### Classification Report")
        report = classification_report(
            y_test, pred, target_names=class_names, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    if model_tab_choice == "Decision Tree":
        st.markdown("#### Decision Tree Visualization")
        fig5, ax5 = plt.subplots(figsize=(14, 8))
        plot_tree(dt, feature_names=FEATURES, class_names=class_names, filled=True, rounded=True, ax=ax5)
        st.pyplot(fig5)
        buf = io.BytesIO()
        fig5.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        st.download_button("⬇️ Download tree as PNG", buf.getvalue(), file_name="decision_tree.png", mime="image/png")

        st.markdown("#### Feature Importance")
        importance = pd.DataFrame(
            {"Feature": FEATURES, "Importance": dt.feature_importances_}
        ).sort_values("Importance", ascending=False)
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        ax6.barh(importance["Feature"], importance["Importance"], color="#5cb85c")
        ax6.invert_yaxis()
        st.pyplot(fig6)
    else:
        st.markdown("#### Model Coefficients")
        coef_df = pd.DataFrame(lr.coef_, columns=FEATURES, index=class_names)
        st.dataframe(coef_df.round(3), use_container_width=True)
        st.caption("One row of coefficients per class (one-vs-rest view of the multinomial model).")

# ----------------------------------------------------------------------------
# Tab 4: Compare Models
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Decision Tree vs Logistic Regression")
    dt_accuracy = accuracy_score(y_test, dt_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    c1, c2 = st.columns(2)
    c1.metric("Decision Tree Accuracy", f"{dt_accuracy*100:.1f}%")
    c2.metric("Logistic Regression Accuracy", f"{lr_accuracy*100:.1f}%")

    fig7, ax7 = plt.subplots(figsize=(4, 3))
    ax7.bar(["Decision Tree", "Logistic Regression"], [dt_accuracy, lr_accuracy], color=["#5cb85c", "#5bc0de"])
    ax7.set_ylabel("Accuracy")
    ax7.set_ylim(0, 1)
    st.pyplot(fig7)

    if dt_accuracy > lr_accuracy:
        st.success("🏆 Decision Tree performed better.")
    elif lr_accuracy > dt_accuracy:
        st.success("🏆 Logistic Regression performed better.")
    else:
        st.info("🤝 Both models performed equally.")

# ----------------------------------------------------------------------------
# Tab 5: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### About this app

        This Streamlit app is an interactive wrapper around the lab notebook
        **"Waste Type Classification for Recycling using Decision Tree and Logistic
        Regression"**. It reproduces the same pipeline:

        1. Load `waste_data.csv` (upload it in the sidebar) with features
           `Weight`, `Moisture`, `Hardness`, `Magnetic`, `Biodegradable`
           and target `WasteType` (e.g. Plastic, Metal, Paper, Glass, Organic).
        2. Encode the target labels with `LabelEncoder`.
        3. Split into train/test sets (80/20 by default, matching the notebook).
        4. Train a `DecisionTreeClassifier` and a `LogisticRegression` model.
        5. Evaluate each with accuracy, a classification report, and a confusion matrix.
        6. Compare the two models' accuracy.
        7. Predict the waste type of a new sample interactively or in batch.

        **Note on demo data** — the original `waste_data.csv` wasn't provided alongside
        the lab document. If you don't upload it, the app falls back to a synthetic
        dataset with the same column schema and roughly realistic per-category
        tendencies, so the app is still fully functional; upload your real CSV in the
        sidebar for accurate results.
        """
    )
