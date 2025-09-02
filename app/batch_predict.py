import pickle
from pathlib import Path
from darts.models import TFTModel
from app.utils import load_and_preprocess, scale_series
import pandas as pd

def batch_predict(df, model_dir="gs://my-bucket/tft_models/latest/", forecast_horizon=4):
    series, covariates = load_and_preprocess(df)

    # Load model + scalers from GCS (download locally first)
    import tempfile
    from google.cloud import storage

    tmp_dir = tempfile.TemporaryDirectory()
    storage_client = storage.Client()
    bucket_name = model_dir.split("/")[2]
    prefix = "/".join(model_dir.split("/")[3:])

    model_local_path = Path(tmp_dir.name) / "tft_model.pth.tar"
    scalers_local_path = Path(tmp_dir.name) / "scalers.pkl"

    bucket = storage_client.bucket(bucket_name)
    bucket.blob(prefix + "tft_model.pth.tar").download_to_filename(model_local_path)
    bucket.blob(prefix + "scalers.pkl").download_to_filename(scalers_local_path)

    model = TFTModel.load(model_local_path)
    with open(scalers_local_path, "rb") as f:
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

