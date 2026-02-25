# Climate-Resilient Agricultural Yield Prediction

A Streamlit-based decision support system for climate-smart agriculture.

This project predicts:
- **Crop yield (kg/ha)** using climate, soil, water, and air-quality features
- **Climate resilience level** (`High`, `Medium`, `Low`) for farming conditions

The repository also includes multiple Jupyter notebooks for EDA and model training workflows.

## Features

- Interactive input form (sliders + select boxes) in Streamlit
- Yield prediction from a trained regression model
- Resilience classification from a trained classifier
- Automatic `Climate_Stress_Index` calculation from stress variables:
  - `Heatwave_Days`
  - `Dry_Spell_Count`
  - `Temp_Anomaly`

## Tech Stack

- Python 3.10+
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── cra_yield_prediction_model.ipynb
├── final_eda.ipynb
├── final_yield_prediction_model_RandomForestRegressor.ipynb
├── resilient_classifier.ipynb
└── README.md
```

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd cliimate_reseleant_agri_yeild_prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Required Model Files

`app.py` expects trained model artifacts inside a `models/` directory.

Create this folder and place these files exactly:

```text
models/
├── CRA_Final_Yield_Prediction_Model.joblib
└── CRA_Final_Resilient_Classifier_Model.joblib
```

> Note: Without these files, the app will fail at startup when loading models.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How the App Works

1. User provides environmental and agricultural inputs from the sidebar.
2. The app computes `Climate_Stress_Index` from normalized stress factors.
3. Yield model predicts expected crop yield.
4. Resilience model predicts resilience class.
5. Results are displayed as metrics with interpretation hints.

## Notebooks

- `final_eda.ipynb` – Exploratory data analysis
- `cra_yield_prediction_model.ipynb` – Yield model experimentation/training
- `final_yield_prediction_model_RandomForestRegressor.ipynb` – Final yield model workflow
- `resilient_classifier.ipynb` – Resilience classification workflow

## Troubleshooting

- **`FileNotFoundError` for model files**:
  Ensure `models/` exists and contains both `.joblib` files with exact names.
- **Streamlit command not found**:
  Activate your virtual environment and reinstall requirements.
- **Version conflicts**:
  Use a clean virtual environment and reinstall from `requirements.txt`.

## Future Improvements

- Add model/version metadata display in UI
- Add input validation and data export
- Add deployment config (Docker/Cloud)
- Add automated tests for feature alignment and prediction pipeline
