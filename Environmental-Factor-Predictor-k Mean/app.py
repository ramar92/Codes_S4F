"""
Environmental Factors K-Means Clustering — Streamlit App
==========================================================
UI + deployment file for the K-Means clustering model built in
`1__k-means_clustering.ipynb`.

Dataset: environmental_factors.csv
Columns: temperature, humidity, wind_speed, carbon_emissions,
         solar_irradiance, pollution_level

Run with:
    streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Environmental Factors — K-Means Clustering",
    page_icon="🌍",
    layout="wide",
)

DEFAULT_FILE = "environmental_factors.csv"
FEATURE_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "carbon_emissions",
    "solar_irradiance",
    "pollution_level",
]

st.title("🌍 Environmental Factors — K-Means Clustering")
st.caption(
    "Explore clusters found in environmental sensor data and classify new "
    "readings using a K-Means model."
)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)


@st.cache_data
def compute_elbow(data_scaled, max_k):
    inertias = []
    for kk in range(1, max_k + 1):
        km = KMeans(n_clusters=kk, random_state=42, n_init=10)
        km.fit(data_scaled)
        inertias.append(km.inertia_)
    return inertias


@st.cache_resource
def fit_kmeans(data_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data_scaled)
    return km, labels


# --------------------------------------------------------------------------
# Sidebar — data source
# --------------------------------------------------------------------------
st.sidebar.header("1. Data")

df = None
if os.path.exists(DEFAULT_FILE):
    df = load_data(DEFAULT_FILE)
    st.sidebar.success(f"Loaded bundled '{DEFAULT_FILE}' ({len(df)} rows).")
else:
    uploaded = st.sidebar.file_uploader(
        f"Upload '{DEFAULT_FILE}'", type=["csv"]
    )
    if uploaded is not None:
        df = load_data(uploaded)
        st.sidebar.success(f"Loaded {len(df)} rows from uploaded file.")

if df is None:
    st.warning(f"Please upload '{DEFAULT_FILE}' to continue.")
    st.stop()

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

data = df[FEATURE_COLS].dropna().reset_index(drop=True)

with st.expander("Preview data", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.write(f"Rows: {len(data)}")

# --------------------------------------------------------------------------
# Scale data
# --------------------------------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------------------------------------------------------
# Sidebar — model settings
# --------------------------------------------------------------------------
st.sidebar.header("2. Model settings")
max_k = min(10, len(data) - 1)
k = st.sidebar.slider("Number of clusters (k)", 2, max_k, value=min(7, max_k))
show_elbow = st.sidebar.checkbox("Show elbow method chart", value=True)

# --------------------------------------------------------------------------
# Elbow method
# --------------------------------------------------------------------------
if show_elbow:
    st.subheader("📉 Elbow Method")
    inertias = compute_elbow(data_scaled, max_k)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, max_k + 1), inertias, marker="o")
    ax.axvline(k, color="red", linestyle="--", label=f"Selected k={k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Fit K-Means
# --------------------------------------------------------------------------
kmeans, labels = fit_kmeans(data_scaled, k)
data_clustered = data.copy()
data_clustered["cluster"] = labels

sil_score = (
    silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else float("nan")
)

col1, col2 = st.columns(2)
col1.metric("Clusters (k)", k)
col2.metric("Silhouette Score", f"{sil_score:.3f}")

# --------------------------------------------------------------------------
# Cluster visualization (PCA to 2D)
# --------------------------------------------------------------------------
st.subheader("🗺️ Cluster Visualization (PCA projection)")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(data_scaled)
centers_2d = pca.transform(kmeans.cluster_centers_)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=15, alpha=0.6)
ax2.scatter(
    centers_2d[:, 0], centers_2d[:, 1],
    c="black", marker="X", s=200, label="Centroids"
)
ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")
ax2.set_title("Clusters (2D PCA projection)")
ax2.legend()
st.pyplot(fig2)

# --------------------------------------------------------------------------
# Cluster profiles
# --------------------------------------------------------------------------
st.subheader("📊 Cluster Profiles (feature means)")
profile = data_clustered.groupby("cluster")[FEATURE_COLS].mean().round(2)
profile["count"] = data_clustered.groupby("cluster").size()
st.dataframe(profile, use_container_width=True)

with st.expander("Show full clustered data"):
    st.dataframe(data_clustered, use_container_width=True)
    st.download_button(
        "Download clustered data as CSV",
        data=data_clustered.to_csv(index=False).encode("utf-8"),
        file_name="clustered_environmental_data.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# Predict cluster for a new data point
# --------------------------------------------------------------------------
st.subheader("🔮 Predict Cluster for New Data")
st.write("Enter environmental readings to see which cluster they belong to.")

pred_cols = st.columns(3)
input_values = {}
for i, feat in enumerate(FEATURE_COLS):
    col = pred_cols[i % 3]
    lo, hi = float(data[feat].min()), float(data[feat].max())
    default = float(data[feat].mean())
    input_values[feat] = col.slider(
        feat.replace("_", " ").title(),
        min_value=round(lo, 2),
        max_value=round(hi, 2),
        value=round(default, 2),
    )

if st.button("Predict Cluster", type="primary"):
    new_point = pd.DataFrame([input_values])[FEATURE_COLS]
    new_scaled = scaler.transform(new_point)
    pred_cluster = int(kmeans.predict(new_scaled)[0])

    st.success(f"This data point belongs to **Cluster {pred_cluster}**")
    st.write("Cluster profile it belongs to:")
    st.dataframe(profile.loc[[pred_cluster]], use_container_width=True)

    new_2d = pca.transform(new_scaled)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=15, alpha=0.4)
    ax3.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=150)
    ax3.scatter(
        new_2d[:, 0], new_2d[:, 1],
        c="red", marker="*", s=400, edgecolors="black", label="New point"
    )
    ax3.set_title("New point relative to existing clusters")
    ax3.legend()
    st.pyplot(fig3)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • K-Means • scikit-learn")
