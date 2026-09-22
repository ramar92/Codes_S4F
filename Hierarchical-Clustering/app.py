"""
Deployment / UI app for the Hierarchical Clustering analysis
(Environmental & Socioeconomic Factors).

This script does NOT redo the exploratory work from the notebook
(dendrogram tuning, cluster-count selection, etc.). It simply takes the
final, already-decided pipeline from `2__Hierarchical_Clustering_.ipynb`
(StandardScaler -> AgglomerativeClustering, n_clusters=7, linkage='ward')
and wraps it in a Streamlit app so users can:

  1. Explore the existing clusters (profiles, distribution, scatter plot).
  2. Enter a new country's stats and see which cluster it's closest to.

Note: AgglomerativeClustering has no native `.predict()` for unseen data,
so new points are assigned to the nearest existing cluster centroid
(in scaled feature space) — the standard way to "deploy" a hierarchical
clustering result.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

DATA_PATH = "environmental_socioeconomic.csv"
FEATURES = ["co2_emissions", "waste_production", "gdp", "population"]
N_CLUSTERS = 3 # decided in the notebook via the dendrogram

st.set_page_config(
    page_title="Environmental & Socioeconomic Clustering",
    page_icon="🌍",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached pipeline: fit once, reuse across interactions
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def fit_pipeline(df: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric="euclidean", linkage="ward"
    )
    labels = model.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    # Centroids in scaled space, used to assign new/unseen points
    centroids = np.array(
        [X_scaled[labels == c].mean(axis=0) for c in range(N_CLUSTERS)]
    )

    result = df.copy()
    result["cluster"] = labels
    return result, scaler, centroids, sil_score


def assign_cluster(new_point: dict, scaler: StandardScaler, centroids: np.ndarray) -> int:
    """Assign a new record to the nearest cluster centroid (scaled space)."""
    x = pd.DataFrame([new_point], columns=FEATURES)
    x_scaled = scaler.transform(x)[0]
    distances = np.linalg.norm(centroids - x_scaled, axis=1)
    return int(np.argmin(distances))


# ---------------------------------------------------------------------
# Load data + fitted pipeline
# ---------------------------------------------------------------------
try:
    data = load_data(DATA_PATH)
except FileNotFoundError:
    st.warning(f"Couldn't find `{DATA_PATH}` next to this script. Upload it below.")
    uploaded = st.file_uploader("Upload environmental_socioeconomic.csv", type="csv")
    if uploaded is None:
        st.stop()
    data = pd.read_csv(uploaded)

clustered_data, scaler, centroids, sil_score = fit_pipeline(data)

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🌍 Clustering App")
page = st.sidebar.radio("Go to", ["Overview & Clusters", "Predict New Data Point"])
st.sidebar.markdown("---")
st.sidebar.metric("Number of clusters", N_CLUSTERS)
st.sidebar.metric("Silhouette score", f"{sil_score:.3f}")

# ---------------------------------------------------------------------
# Page 1: Overview & Clusters
# ---------------------------------------------------------------------
if page == "Overview & Clusters":
    st.title("Hierarchical Clustering — Overview")
    st.caption(
        "Environmental & socioeconomic factors grouped with Agglomerative "
        "Clustering (Ward linkage, 7 clusters)."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Clusters by CO₂ emissions vs. waste production")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x="co2_emissions",
            y="waste_production",
            hue="cluster",
            data=clustered_data,
            palette="viridis",
            s=60,
            alpha=0.7,
            edgecolor="k",
            ax=ax,
        )
        ax.set_xlabel("CO₂ Emissions")
        ax.set_ylabel("Waste Production")
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig)

    with col2:
        st.subheader("Cluster sizes")
        counts = clustered_data["cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax2)
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    st.subheader("Cluster profiles (average feature values)")
    profile = clustered_data.groupby("cluster")[FEATURES].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    with st.expander("Show raw clustered data"):
        st.dataframe(clustered_data, use_container_width=True)

# ---------------------------------------------------------------------
# Page 2: Predict New Data Point
# ---------------------------------------------------------------------
else:
    st.title("Assign a New Record to a Cluster")
    st.caption(
        "Enter values for a new country/region and see which existing "
        "cluster it's closest to (nearest centroid in scaled feature space)."
    )

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            co2 = st.number_input(
                "CO₂ emissions",
                min_value=0.0,
                value=float(data["co2_emissions"].mean()),
            )
            waste = st.number_input(
                "Waste production",
                min_value=0.0,
                value=float(data["waste_production"].mean()),
            )
        with c2:
            gdp = st.number_input(
                "GDP", min_value=0.0, value=float(data["gdp"].mean())
            )
            population = st.number_input(
                "Population",
                min_value=0.0,
                value=float(data["population"].mean()),
            )
        submitted = st.form_submit_button("Predict cluster")

    if submitted:
        new_point = {
            "co2_emissions": co2,
            "waste_production": waste,
            "gdp": gdp,
            "population": population,
        }
        cluster_id = assign_cluster(new_point, scaler, centroids)

        st.success(f"This record is closest to **Cluster {cluster_id}**")

        st.subheader(f"Cluster {cluster_id} profile vs. your input")
        profile_row = clustered_data[clustered_data["cluster"] == cluster_id][
            FEATURES
        ].mean()
        comparison = pd.DataFrame(
            {
                "Your input": new_point,
                f"Cluster {cluster_id} average": profile_row,
            }
        )
        st.dataframe(comparison.round(2), use_container_width=True)

        st.subheader("Where it lands on the map")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x="co2_emissions",
            y="waste_production",
            hue="cluster",
            data=clustered_data,
            palette="viridis",
            s=50,
            alpha=0.5,
            edgecolor="k",
            ax=ax3,
        )
        ax3.scatter(
            co2,
            waste,
            color="red",
            s=250,
            marker="*",
            edgecolor="black",
            label="New point",
            zorder=5,
        )
        ax3.set_xlabel("CO₂ Emissions")
        ax3.set_ylabel("Waste Production")
        ax3.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig3)
