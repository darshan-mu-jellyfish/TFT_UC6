import pickle
from pathlib import Path
from darts.models import TFTModel
from app.utils import load_and_preprocess, scale_series

def train_model(df, model_dir="models/", forecast_horizon=4):
    series, covariates = load_and_preprocess(df)
    series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series, covariates)

    # Train single global TFT across series
    model = TFTModel(
        input_chunk_length=24,
        output_chunk_length=forecast_horizon,
        hidden_size=16,
        n_epochs=10,
        random_state=42,
    )

    model.fit(series_scaled, past_covariates=covs_scaled, verbose=True)

    # Save model + scalers
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(model_dir) / "tft_model.pth.tar")

    with open(Path(model_dir) / "scalers.pkl", "wb") as f:
        pickle.dump((scaler_y, scaler_x), f)

    return model
