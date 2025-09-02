import pickle
from pathlib import Path
import pandas as pd
from darts.models import TFTModel
from app.utils import load_and_preprocess

def batch_predict(df, model_dir="models/", forecast_horizon=4):
    series, covariates = load_and_preprocess(df)

    # Load model + scalers
    model = TFTModel.load(Path(model_dir) / "tft_model.pth.tar")
    with open(Path(model_dir) / "scalers.pkl", "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    series_scaled = [scaler_y.transform(s) for s in series]
    covs_scaled = [scaler_x.transform(c) for c in covariates]

    forecasts = []
    for sid, ts, cov in zip(df["series_id_encoded"].unique(), series_scaled, covs_scaled):
        pred = model.predict(forecast_horizon, past_covariates=cov)
        pred = scaler_y.inverse_transform(pred)
        f_df = pred.pd_dataframe().reset_index()
        f_df["series_id_encoded"] = sid
        forecasts.append(f_df)

    return pd.concat(forecasts, ignore_index=True)
