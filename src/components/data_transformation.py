import sys
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config, save_object


class DataTransformation:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

    def check_stationarity(self, series, label="series"):
        """
        Runs the Augmented Dickey-Fuller test.
        Returns True if stationary (p-value < 0.05), else False.
        """
        try:
            result = adfuller(series.dropna())
            adf_stat, p_value = result[0], result[1]
            is_stationary = p_value < 0.05

            logging.info(
                f"ADF test on {label} -> statistic: {adf_stat:.4f}, p-value: {p_value:.4f}, "
                f"stationary: {is_stationary}"
            )
            return is_stationary, p_value

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, series, exog_df=None):
        """
        1. Optionally log-transforms the target series (exog columns are left untouched —
           they're on their own natural scales and SARIMAX doesn't require them to match).
        2. Runs ADF test on the target (informational — ARIMA's 'd' term handles
           differencing internally, so we don't manually difference here).
        3. Splits target AND exog chronologically into train/test using the SAME split
           index, so rows stay aligned (no shuffling, no random_state).
        """
        logging.info("Starting data transformation")
        try:
            trans_cfg = self.config["transformation"]
            split_cfg = self.config["split"]

            if exog_df is None:
                exog_df = pd.DataFrame(index=series.index)

            working_series = series.copy()

            if trans_cfg.get("apply_log_transform", False):
                if (working_series <= 0).any():
                    logging.info("Log transform requested but series has non-positive values — skipping.")
                else:
                    working_series = np.log(working_series)
                    logging.info("Applied log transform to series.")

            is_stationary, p_value = self.check_stationarity(working_series, label="target series")

            test_size = split_cfg.get("test_size", 0.2)
            split_idx = int(len(working_series) * (1 - test_size))

            train_series = working_series.iloc[:split_idx]
            test_series = working_series.iloc[split_idx:]

            train_exog = exog_df.iloc[:split_idx]
            test_exog = exog_df.iloc[split_idx:]

            logging.info(
                f"Chronological split complete. Train: {len(train_series)} rows, "
                f"Test: {len(test_series)} rows. Exog columns: {list(exog_df.columns)}"
            )

            transform_info = {
                "log_transform_applied": trans_cfg.get("apply_log_transform", False),
                "is_stationary": is_stationary,
                "adf_p_value": p_value,
                "exog_columns": list(exog_df.columns),
            }

            artifacts_cfg = self.config["artifacts"]
            save_object(artifacts_cfg["transform_info_path"], transform_info)

            train_series.to_csv(artifacts_cfg["train_data_path"], header=True)
            test_series.to_csv(artifacts_cfg["test_data_path"], header=True)

            if not exog_df.empty:
                train_exog.to_csv(artifacts_cfg["train_exog_path"], header=True)
                test_exog.to_csv(artifacts_cfg["test_exog_path"], header=True)

            return train_series, test_series, train_exog, test_exog, transform_info

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    series, exog_df = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_series, test_series, train_exog, test_exog, info = transformer.initiate_data_transformation(
        series, exog_df
    )

    print("Train size:", len(train_series))
    print("Test size:", len(test_series))
    print("Transform info:", info)
