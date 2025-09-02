import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from google.cloud import bigquery
import pandas as pd

def load_data_from_bq(project_id: str, dataset: str, table: str, where: str = None):
    """
    Load data from BigQuery into pandas DataFrame.
    """
    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT series_id_encoded, timestamp, sales, on_promotion, price, category_encoded
        FROM `{project_id}.{dataset}.{table}`
    """
    if where:
        query += f" WHERE {where}"

    df = client.query(query).to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_and_preprocess(df: pd.DataFrame):
    """
    Convert raw dataframe into Darts TimeSeries objects.
    Schema: series_id_encoded, timestamp, sales, on_promotion, price, category_encoded
    """
    series_list = []
    covariates_list = []

    for series_id, group in df.groupby("series_id_encoded"):
        group = group.sort_values("timestamp")

        # Target
        ts = TimeSeries.from_dataframe(
            group,
            time_col="timestamp",
            value_cols="sales",
            freq="W"  # adjust if daily/hourly
        )
        series_list.append(ts)

        # Covariates (dynamic)
        cov = TimeSeries.from_dataframe(
            group,
            time_col="timestamp",
            value_cols=["on_promotion", "price"],
            freq="W"
        )
        covariates_list.append(cov)

    return series_list, covariates_list


def scale_series(series_list, covariates_list):
    """Apply scaling"""
    scaler_y = Scaler()
    scaler_x = Scaler()

    series_scaled = [scaler_y.fit_transform(s) for s in series_list]
    covs_scaled = [scaler_x.fit_transform(c) for c in covariates_list]

    return series_scaled, covs_scaled, scaler_y, scaler_x
