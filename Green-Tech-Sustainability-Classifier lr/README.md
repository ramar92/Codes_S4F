# Green-Tech Sustainability Classifier (Streamlit)

An interactive Streamlit app built from `3___Logistic_regression_.ipynb`. It trains a
Logistic Regression model on a synthetic "green tech" dataset and lets you:

- Predict sustainability live with sliders
- Score a batch of scenarios by uploading a CSV
- Explore the dataset (distributions, correlations, class balance)
- Inspect model performance (accuracy, confusion matrix, ROC/AUC, coefficients)

## Files

```
streamlit_app/
├── app.py             # the Streamlit app
├── requirements.txt   # Python dependencies
└── README.md          # this file
```

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
docker build -t sustainability-app .
docker run -p 8501:8501 sustainability-app
```

Push the image to any container registry (Docker Hub, ECR, GCR) and deploy it on
Render, Railway, AWS App Runner, Google Cloud Run, Azure Container Apps, etc.

## Notes

- The app regenerates the exact synthetic dataset from the notebook using the same
  random seed (42) by default, or you can upload your own CSV with columns:
  `carbon_emissions, energy_output, renewability_index, cost_efficiency, sustainability`.
- The model is retrained inside the app (cached with `st.cache_resource`) so it always
  matches the currently selected dataset/settings — no separate `.pkl` file is required,
  though you can still export one with `joblib.dump(model, "lrmodel_sustainable.pkl")`
  if you want to reuse it elsewhere.
