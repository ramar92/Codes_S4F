# Waste Type Classification (Streamlit)

An interactive Streamlit app built from the lab
**"Waste Type Classification for Recycling using Decision Tree and Logistic Regression"**.
It trains both a `DecisionTreeClassifier` and a `LogisticRegression` model to classify
waste items (Plastic, Metal, Paper, Glass, Organic) and lets you:

- Predict the waste type live from Weight, Moisture, Hardness, Magnetic, Biodegradable
- Score a batch of items by uploading a CSV
- Explore the dataset (class balance, feature distributions, correlations)
- Inspect model performance for either model (accuracy, confusion matrix, classification
  report, tree visualization / feature importance / coefficients)
- Directly compare Decision Tree vs Logistic Regression accuracy

## Files

```
waste_lab/
├── Lab_2_Waste_Type_Classification.ipynb   # the notebook (from the lab document)
├── app.py                                  # the Streamlit app
├── requirements.txt                        # Python dependencies
└── README.md                               # this file
```

## ⚠️ About the dataset

The lab loads `waste_data.csv`, which wasn't included with the lab document — only the
code. The app therefore lets you:

- **Upload the real CSV** in the sidebar (columns: `Weight`, `Moisture`, `Hardness`,
  `Magnetic`, `Biodegradable`, `WasteType`) — recommended, or
- **Use the built-in demo dataset**, a synthetic stand-in with realistic per-category
  tendencies (e.g. Metal = heavy + magnetic, Organic = high moisture + biodegradable),
  clearly labeled in the UI, so the app is fully functional even without the original file.

## 1. Run it locally

```bash
# (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## 2. Deploy for free — Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (must contain `app.py` and `requirements.txt`).
   Optionally include your real `waste_data.csv` in the repo if you want a default
   dataset baked in (then add a small code tweak to auto-load it).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to `app.py`.
4. Click **Deploy**. You'll get a shareable URL like
   `https://<your-app-name>.streamlit.app`.

Any time you push new commits to the repo, the app redeploys automatically.

## 3. Deploy on Hugging Face Spaces (alternative)

1. Create a new Space at https://huggingface.co/new-space, choosing **Streamlit** as the SDK.
2. Upload `app.py` and `requirements.txt` (or push via git — Spaces are git repos).
3. The Space builds and serves the app automatically at
   `https://huggingface.co/spaces/<username>/<space-name>`.

## 4. Deploy with Docker (any cloud VM / container service)

Create a `Dockerfile` alongside `app.py`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then build and run:

```bash
docker build -t waste-classifier-app .
docker run -p 8501:8501 waste-classifier-app
```

Push the image to any container registry (Docker Hub, ECR, GCR) and deploy it on
Render, Railway, AWS App Runner, Google Cloud Run, Azure Container Apps, etc.

## Notes

- Both models are retrained live (cached with `st.cache_resource`) whenever you change
  the dataset, test split size, or random state.
- The Predict tab lets you toggle between Decision Tree and Logistic Regression to see
  how each one classifies the same input.
- The Model Performance tab shows the Decision Tree visualization and feature
  importances, or the Logistic Regression coefficients, depending on which model is
  selected.
