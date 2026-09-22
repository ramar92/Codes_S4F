"""
Smart City Energy Consumption — K-Means Clustering App
=========================================================
UI + deployment file for the model built in
`Lab_3_-_Clustering_Energy_Consumption_Patterns_for_Smart_Cities.ipynb`.

Dataset: energy_data.csv
Columns: timestamp, location, energy_consumption, temperature

Clustering features: energy_consumption, temperature

Run with:
    streamlit run app.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart City Energy Consumption Clustering",
    page_icon="⚡",
    layout="wide",
)

DEFAULT_FILE = "energy_data.csv"
FEATURE_COLS = ["energy_consumption", "temperature"]

st.title("⚡ Smart City Energy Consumption — K-Means Clustering")
st.caption(
    "Explore clusters of energy consumption patterns across temperature "
    "conditions, and classify new readings using a K-Means model."
)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna()
    return df


@st.cache_data
def compute_elbow(data_scaled, max_k):
    wcss = []
    for i in range(1, max_k + 1):
        km = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
        km.fit(data_scaled)
        wcss.append(km.inertia_)
    return wcss


@st.cache_resource
def fit_kmeans(data_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
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
    uploaded = st.sidebar.file_uploader(f"Upload '{DEFAULT_FILE}'", type=["csv"])
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

if "location" in df.columns:
    locations = sorted(df["location"].dropna().unique().tolist())
    selected_locations = st.sidebar.multiselect(
        "Filter by location", locations, default=locations
    )
    df = df[df["location"].isin(selected_locations)]

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Rows after filtering: {len(df)}")

data = df[FEATURE_COLS].reset_index(drop=True)

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
k = st.sidebar.slider("Number of clusters (k)", 2, max_k, value=min(3, max_k))
show_elbow = st.sidebar.checkbox("Show elbow method chart", value=True)

# --------------------------------------------------------------------------
# Elbow method
# --------------------------------------------------------------------------
if show_elbow:
    st.subheader("📉 Elbow Method")
    wcss = compute_elbow(data_scaled, max_k)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, max_k + 1), wcss, marker="o", linestyle="-")
    ax.axvline(k, color="red", linestyle="--", label=f"Selected k={k}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Fit K-Means
# --------------------------------------------------------------------------
kmeans, labels = fit_kmeans(data_scaled, k)
data_clustered = df.copy().reset_index(drop=True)
data_clustered["cluster"] = labels

sil_score = (
    silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else float("nan")
)

col1, col2 = st.columns(2)
col1.metric("Clusters (k)", k)
col2.metric("Silhouette Score", f"{sil_score:.2f}")

# --------------------------------------------------------------------------
# Cluster visualization (native 2D — energy_consumption vs temperature)
# --------------------------------------------------------------------------
st.subheader("🗺️ Cluster Visualization")
fig2, ax2 = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(
    x="energy_consumption",
    y="temperature",
    hue="cluster",
    data=data_clustered,
    palette="viridis",
    s=50,
    ax=ax2,
)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax2.scatter(
    centers[:, 0], centers[:, 1],
    c="red", marker="X", s=250, label="Centroids", edgecolors="black"
)
ax2.set_title("Clusters of Energy Consumption Patterns")
ax2.set_xlabel("Energy Consumption")
ax2.set_ylabel("Temperature")
ax2.legend(title="Cluster")
st.pyplot(fig2)

# --------------------------------------------------------------------------
# Cluster profiles
# --------------------------------------------------------------------------
st.subheader("📊 Cluster Profiles")
profile = data_clustered.groupby("cluster")[FEATURE_COLS].mean().round(2)
profile["count"] = data_clustered.groupby("cluster").size()
st.dataframe(profile, use_container_width=True)

if "location" in data_clustered.columns:
    st.write("Location distribution per cluster:")
    loc_dist = pd.crosstab(data_clustered["cluster"], data_clustered["location"])
    st.dataframe(loc_dist, use_container_width=True)

with st.expander("Show full clustered data"):
    st.dataframe(data_clustered, use_container_width=True)
    st.download_button(
        "Download clustered data as CSV",
        data=data_clustered.to_csv(index=False).encode("utf-8"),
        file_name="clustered_energy_data.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# Predict cluster for a new reading
# --------------------------------------------------------------------------
st.subheader("🔮 Predict Cluster for New Reading")
st.write("Enter an energy consumption and temperature value to classify it.")

col_a, col_b = st.columns(2)
energy_val = col_a.slider(
    "Energy Consumption",
    float(data["energy_consumption"].min()),
    float(data["energy_consumption"].max()),
    float(data["energy_consumption"].mean()),
)
temp_val = col_b.slider(
    "Temperature",
    float(data["temperature"].min()),
    float(data["temperature"].max()),
    float(data["temperature"].mean()),
)

if st.button("Predict Cluster", type="primary"):
    new_point = pd.DataFrame([[energy_val, temp_val]], columns=FEATURE_COLS)
    new_scaled = scaler.transform(new_point)
    pred_cluster = int(kmeans.predict(new_scaled)[0])

    st.success(f"This reading belongs to **Cluster {pred_cluster}**")
    st.write("Cluster profile it belongs to:")
    st.dataframe(profile.loc[[pred_cluster]], use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        x="energy_consumption", y="temperature", hue="cluster",
        data=data_clustered, palette="viridis", s=40, alpha=0.5, ax=ax3
    )
    ax3.scatter(
        energy_val, temp_val,
        c="red", marker="*", s=500, edgecolors="black", label="New reading"
    )
    ax3.set_title("New reading relative to existing clusters")
    ax3.set_xlabel("Energy Consumption")
    ax3.set_ylabel("Temperature")
    ax3.legend()
    st.pyplot(fig3)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • K-Means • scikit-learn")
