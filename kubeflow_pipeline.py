import kfp
from kfp.v2 import dsl
from kfp.v2.google.client import AIPlatformClient

# ================== PARAMETERS ==================
PROJECT_ID = "tr-tech-dds-dev"            # Your GCP project
REGION = "europe-west1"                   # Region for Vertex AI
STAGING_BUCKET = "gs://tr-tech-dds-dev-tft_test"  # Staging bucket
IMAGE_URI = "europe-west1-docker.pkg.dev/tr-tech-dds-dev/forecasting/tft_test:v5.0"

BQ_DATASET = "test_datafeed_laredoute"
BQ_TABLE = "test_table_daily"

MODEL_OUTPUT_PATH = f"{STAGING_BUCKET}/tft_model_output"
PREDICTION_OUTPUT_PATH = f"{STAGING_BUCKET}/tft_predictions"

# ================================================

def create_custom_job_op(mode: str, output_path: str, where: str = None):
    """
    Returns a Kubeflow pipeline container op that runs your Docker container.
    """
    from kfp.v2.dsl import component, Input, Output, Dataset, Model

    @dsl.container_component
    def _op():
        return dsl.ContainerSpec(
            image=IMAGE_URI,
            command=[
                "python3",
                "/app/pipeline_forecast.py",
            ],
            args=[
                "--mode", mode,
                "--data_source", "bq",
                "--project_id", PROJECT_ID,
                "--dataset", BQ_DATASET,
                "--table", BQ_TABLE,
                "--where", where if where else "",
                "--output_path", output_path,
            ],
        )

    return _op()


# ================== PIPELINE DEFINITION ==================
@dsl.pipeline(
    name="tft-forecasting-pipeline",
    description="Kubeflow pipeline for Darts TFT training and batch prediction"
)
def tft_pipeline():
    # ----- Training step -----
    train_step = create_custom_job_op(
        mode="train",
        output_path=MODEL_OUTPUT_PATH,
        where="timestamp > '2023-01-01'"
    )

    # ----- Batch prediction step -----
    predict_step = create_custom_job_op(
        mode="predict",
        output_path=PREDICTION_OUTPUT_PATH
    )

    # Make batch prediction dependent on training
    predict_step.after(train_step)


# ================== PIPELINE SUBMISSION ==================
if __name__ == "__main__":
    client = AIPlatformClient(project_id=PROJECT_ID, region=REGION)
    
    # Compile pipeline
    pipeline_root = f"{STAGING_BUCKET}/tft_pipeline_root"
    pipeline_file = "tft_pipeline.json"

    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=tft_pipeline,
        package_path=pipeline_file
    )

    # Submit pipeline
    client.create_run_from_job_spec(
        job_spec_path=pipeline_file,
        pipeline_root=pipeline_root,
        enable_caching=False
    )
