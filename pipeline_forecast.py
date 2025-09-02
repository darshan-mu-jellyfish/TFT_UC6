import argparse
import pandas as pd
from app.train import train_model
from app.batch_predict import batch_predict
from app.utils import load_and_preprocess

from google.cloud import bigquery

def load_bq_data(project_id, dataset, table, where=None):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"
    if where:
        query += f" WHERE {where}"
    df = client.query(query).to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data_source", choices=["bq", "csv"], default="bq")
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--table", type=str)
    parser.add_argument("--where", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--forecast_horizon", type=int, default=4)
    parser.add_argument("--repo_image_uri", type=str, default=None)
    args = parser.parse_args()

    # Load data
    if args.data_source == "bq":
        df = load_bq_data(args.project_id, args.dataset, args.table, args.where)
    else:
        df = pd.read_csv(args.csv_path)

    if args.mode == "train":
        train_model(
            df,
            bucket_name=args.bucket_name,
            forecast_horizon=args.forecast_horizon,
            project_id=args.project_id,
            repo_image_uri=args.repo_image_uri
        )
    elif args.mode == "predict":
        forecasts = batch_predict(
            df,
            model_dir=f"gs://{args.bucket_name}/tft_models/latest/",
            forecast_horizon=args.forecast_horizon
        )
        forecasts.to_csv("predictions.csv", index=False)
        print("✅ Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()

