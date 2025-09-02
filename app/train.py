import pickle
from pathlib import Path
from darts.models import TFTModel
from app.utils import load_and_preprocess, scale_series

def train_model(df,
    bucket_name="my-bucket",
    model_root="tft_models",
    forecast_horizon=4,
    project_id="my-project",
    location="europe-west1",
    repo_image_uri=None
):
    series, covariates = load_and_preprocess(df)
    series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series, covariates)

    # Train single global TFT across series
    model = TFTModel(
        input_chunk_length=24,
        output_chunk_length=forecast_horizon,
        hidden_size=16,
        n_epochs=10,
        random_state=42,
        add_relative_index=True
    )

    # ===== 2. Save locally to /tmp =====
    version = datetime.datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    tmp_dir = Path(f"/tmp/{version}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_path = tmp_dir / "tft_model.pth.tar"
    scalers_path = tmp_dir / "scalers.pkl"

    model.save(str(model_path))
    with open(scalers_path, "wb") as f:
        pickle.dump((scaler_y, scaler_x), f)

    # ===== 3. Upload artifacts to GCS =====
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    gcs_prefix = f"{model_root}/{version}/"

    bucket.blob(gcs_prefix + "tft_model.pth.tar").upload_from_filename(model_path)
    bucket.blob(gcs_prefix + "scalers.pkl").upload_from_filename(scalers_path)

    # Update latest pointer
    bucket.blob(f"{model_root}/latest/tft_model.pth.tar").upload_from_filename(model_path)
    bucket.blob(f"{model_root}/latest/scalers.pkl").upload_from_filename(scalers_path)

    print(f"✅ Model uploaded to gs://{bucket_name}/{gcs_prefix}")

    # ===== 4. Register in Vertex AI =====
    if repo_image_uri:
        aiplatform.init(project=project_id, location=location)
        registered_model = aiplatform.Model.upload(
            display_name="darts-tft-model",
            artifact_uri=f"gs://{bucket_name}/{model_root}/{version}/",
            serving_container_image_uri=repo_image_uri,
        )
        print(f"✅ Registered in Vertex AI: {registered_model.resource_name}")
        return registered_model

    return f"gs://{bucket_name}/{gcs_prefix}"
