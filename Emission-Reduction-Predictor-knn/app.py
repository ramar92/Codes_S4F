import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emission Reduction Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — colorful, modern, "climate tech" theme
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    h1, h2, h3, h4 {
        color: #eafff5 !important;
        font-family: 'Trebuchet MS', sans-serif;
    }
    p, label, span, div {
        color: #e8f7f0;
    }
    .hero-banner {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero-banner h1 {
        color: #05261f !important;
        margin: 0;
        font-size: 2.3rem;
    }
    .hero-banner p {
        color: #06382d !important;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(6px);
    }
    .result-effective {
        background: linear-gradient(120deg, #11998e, #38ef7d);
        color: #06281f;
        padding: 26px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(56,239,125,0.35);
        animation: pulseGreen 1.8s ease-in-out infinite;
    }
    .result-not-effective {
        background: linear-gradient(120deg, #ff5f6d, #ffc371);
        color: #401515;
        padding: 26px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(255,95,109,0.35);
    }
    @keyframes pulseGreen {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1e23 0%, #16323a 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: #06281f;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 26px;
        font-size: 1.05rem;
        transition: 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(56,239,125,0.45);
    }
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.06);
        border-radius: 10px 10px 0 0;
        color: #e8f7f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🌍 Emission Reduction Predictor</h1>
        <p>Powered by a K-Nearest Neighbors model &nbsp;•&nbsp;
        Estimate whether a proposed energy strategy will effectively reduce emissions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# DATA + MODEL (cached)
# ────────────────────────────────────────────────────────────────────────────
DATA_PATH = "emissions_reduction_data.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


@st.cache_resource
def train_model(df, k):
    X = df[["energy_efficiency", "renewable_ratio", "technology_cost"]]
    y = df["emission_reduction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Not Effective", "Effective"], output_dict=True
    )

    return model, scaler, acc, cm, report, X_test, y_test, y_pred


data = load_data(DATA_PATH)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — inputs
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    k = st.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=5, step=1)

    st.markdown("---")
    st.markdown("## 🧮 Input Your Scenario")

    energy_efficiency = st.slider(
        "⚡ Energy Efficiency (%)",
        float(data.energy_efficiency.min()),
        float(data.energy_efficiency.max()),
        float(data.energy_efficiency.mean()),
        step=0.1,
    )
    renewable_ratio = st.slider(
        "♻️ Renewable Energy Ratio",
        0.0,
        1.0,
        float(data.renewable_ratio.mean()),
        step=0.01,
    )
    technology_cost = st.slider(
        "💰 Technology Cost ($)",
        float(data.technology_cost.min()),
        float(data.technology_cost.max()),
        float(data.technology_cost.mean()),
        step=10.0,
    )

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Emission Reduction")

# Train (cached on k)
model, scaler, acc, cm, report, X_test, y_test, y_pred = train_model(data, k)

# ────────────────────────────────────────────────────────────────────────────
# TOP METRIC ROW
# ────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="metric-card"><h3>🎯 Accuracy</h3>
        <h2 style="color:#38ef7d !important;">{acc*100:.2f}%</h2></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="metric-card"><h3>📊 Rows</h3>
        <h2 style="color:#38ef7d !important;">{len(data):,}</h2></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="metric-card"><h3>✅ Effective Rate</h3>
        <h2 style="color:#38ef7d !important;">{data.emission_reduction.mean()*100:.1f}%</h2></div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="metric-card"><h3>🧭 k Value</h3>
        <h2 style="color:#38ef7d !important;">{k}</h2></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Model Performance", "🔍 Explore Data"])

# ---- TAB 1: PREDICTION ----
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Your Scenario")
        input_df = pd.DataFrame(
            {
                "energy_efficiency": [energy_efficiency],
                "renewable_ratio": [renewable_ratio],
                "technology_cost": [technology_cost],
            }
        )
        st.dataframe(input_df.style.format(precision=2), use_container_width=True)

        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    energy_efficiency / data.energy_efficiency.max() * 100,
                    renewable_ratio * 100,
                    100 - (technology_cost / data.technology_cost.max() * 100),
                ],
                theta=["Energy Efficiency", "Renewable Ratio", "Cost Advantage"],
                fill="toself",
                line_color="#38ef7d",
            )
        )
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="white"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=20, b=20),
            height=350,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with right:
        st.markdown("### Prediction Result")
        if predict_clicked:
            scaled_input = scaler.transform(input_df)
            pred = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0]

            if pred == 1:
                st.markdown(
                    f"""<div class="result-effective">✅ EFFECTIVE REDUCTION<br>
                    <span style="font-size:1rem;">Confidence: {proba[1]*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )
                st.balloons()
            else:
                st.markdown(
                    f"""<div class="result-not-effective">⚠️ NOT EFFECTIVE<br>
                    <span style="font-size:1rem;">Confidence: {proba[0]*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            prob_fig = px.bar(
                x=["Not Effective", "Effective"],
                y=proba,
                color=["Not Effective", "Effective"],
                color_discrete_map={"Not Effective": "#ff5f6d", "Effective": "#38ef7d"},
                labels={"x": "Class", "y": "Probability"},
                text=[f"{p*100:.1f}%" for p in proba],
            )
            prob_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
                height=300,
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        else:
            st.info("👈 Adjust the sliders in the sidebar and click **Predict Emission Reduction** to see results here.")

# ---- TAB 2: MODEL PERFORMANCE ----
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Tealgrn",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Effective", "Effective"],
            y=["Not Effective", "Effective"],
        )
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with col2:
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown("### Accuracy vs. Number of Neighbors (k)")
    k_values = list(range(1, 26))
    acc_scores = []
    X = data[["energy_efficiency", "renewable_ratio", "technology_cost"]]
    y = data["emission_reduction"]
    X_train, X_test_k, y_train, y_test_k = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test_k)
    for kv in k_values:
        m = KNeighborsClassifier(n_neighbors=kv)
        m.fit(X_train_s, y_train)
        acc_scores.append(accuracy_score(y_test_k, m.predict(X_test_s)))

    k_fig = px.line(
        x=k_values, y=acc_scores, markers=True,
        labels={"x": "k (Number of Neighbors)", "y": "Accuracy"},
    )
    k_fig.add_vline(x=k, line_dash="dash", line_color="#38ef7d",
                     annotation_text=f"Current k={k}", annotation_font_color="white")
    k_fig.update_traces(line_color="#38ef7d", marker=dict(size=8, color="#11998e"))
    k_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=380,
    )
    st.plotly_chart(k_fig, use_container_width=True)

# ---- TAB 3: EXPLORE DATA ----
with tab3:
    st.markdown("### Dataset Snapshot")
    st.dataframe(data.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Distributions")
        feature_choice = st.selectbox(
            "Choose a feature", ["energy_efficiency", "renewable_ratio", "technology_cost"]
        )
        hist_fig = px.histogram(
            data, x=feature_choice, color="emission_reduction",
            color_discrete_map={0: "#ff5f6d", 1: "#38ef7d"},
            barmode="overlay", nbins=40,
            labels={"emission_reduction": "Effective"},
        )
        hist_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=380,
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.markdown("### Feature Relationships")
        scatter_fig = px.scatter(
            data, x="energy_efficiency", y="renewable_ratio",
            color="emission_reduction", size="technology_cost",
            color_discrete_map={0: "#ff5f6d", 1: "#38ef7d"},
            opacity=0.6,
            labels={"emission_reduction": "Effective"},
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=380,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("### Class Balance")
    pie_fig = px.pie(
        data, names=data["emission_reduction"].map({0: "Not Effective", 1: "Effective"}),
        color=data["emission_reduction"].map({0: "Not Effective", 1: "Effective"}),
        color_discrete_map={"Not Effective": "#ff5f6d", "Effective": "#38ef7d"},
        hole=0.45,
    )
    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380,
    )
    st.plotly_chart(pie_fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Built with Streamlit • KNN Classifier • "
    "Data-driven climate insights 🌱</p>",
    unsafe_allow_html=True,
)
