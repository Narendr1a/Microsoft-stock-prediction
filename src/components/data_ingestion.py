import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config


class DataIngestion:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

    def initiate_data_ingestion(self):
        """
        Reads the raw CSV, parses the date column, sets it as a sorted
        DatetimeIndex, and returns:
            - series: the target column as a pandas Series
            - exog_df: a DataFrame of exogenous columns (empty DataFrame if none configured)
        No shuffling — order matters for time series.
        """
        logging.info("Starting data ingestion")
        try:
            data_cfg = self.config["data"]
            raw_path = data_cfg["raw_data_path"]
            date_col = data_cfg["date_column"]
            target_col = data_cfg["target_column"]
            exog_cols = data_cfg.get("exog_columns", []) or []
            date_format = data_cfg.get("date_format")

            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"Dataset not found at '{raw_path}'. "
                    f"Place your CSV there and update config.yaml if needed."
                )

            df = pd.read_csv(raw_path)
            logging.info(f"Read raw data with shape {df.shape}")

            if date_col not in df.columns:
                raise KeyError(f"date_column '{date_col}' not found in dataset columns: {list(df.columns)}")
            if target_col not in df.columns:
                raise KeyError(f"target_column '{target_col}' not found in dataset columns: {list(df.columns)}")
            missing_exog = [c for c in exog_cols if c not in df.columns]
            if missing_exog:
                raise KeyError(f"exog_columns {missing_exog} not found in dataset columns: {list(df.columns)}")

            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
            if df[date_col].isna().any():
                logging.info("Some dates failed to parse and were dropped.")
                df = df.dropna(subset=[date_col])

            df = df.sort_values(date_col)
            df = df.set_index(date_col)
            df = df[~df.index.duplicated(keep="first")]

            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
                logging.info(f"Inferred and set frequency: {inferred_freq}")
            else:
                logging.info("Could not infer a regular frequency — leaving index as-is.")

            series = df[target_col].astype(float)
            if series.isna().any():
                logging.info("Missing values found in target series — forward/back filling.")
                series = series.ffill().bfill()

            if exog_cols:
                exog_df = df[exog_cols].astype(float)
                if exog_df.isna().any().any():
                    logging.info("Missing values found in exog columns — forward/back filling.")
                    exog_df = exog_df.ffill().bfill()
            else:
                exog_df = pd.DataFrame(index=df.index)

            logging.info(
                f"Data ingestion complete. Series length: {len(series)}, "
                f"exog columns: {list(exog_df.columns)}"
            )
            return series, exog_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    series, exog_df = ingestion.initiate_data_ingestion()
    print(series.head())
    print(f"\nTotal points: {len(series)}")
    if not exog_df.empty:
        print("\nExog features:")
        print(exog_df.head())
