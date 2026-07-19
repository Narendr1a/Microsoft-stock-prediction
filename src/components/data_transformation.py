import sys
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config, save_object

class DataTransformation:
    def __init__(self, config_path):
        self.config = read_config(config_path)

    def check_stationarity(self, series, label = "series"):
        """
#        Runs the Augmented Dickey-Fuller test.
#        Returns True if stationary (p-value < 0.05), else False.
#        """
        try:
            result = adfuller(series.dropna())
            adf_stat, p_value = result[0], result[1]
            is_stationary = p_value < 0.05

            logging.info(
                f"ADF test on {label} -> statistic: {adf_stat: .4f}, p-value: {p_value: .4f}, "
                f"Stationary = {is_stationary}"
            )

            return is_stationary, p_value
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, series):
        """
#        1. Optionally log-transforms the series (for exponential trend / stabilizing variance).
#        2. Runs ADF test to check stationarity (informational — ARIMA's 'd' term handles
#           differencing internally, so we don't manually difference here).
#        3. Splits chronologically into train/test (no shuffling, no random_state).
#        """
        logging.info("Starting data transformation")
        try:
            trans_cfg = self.config["transformation"]
            split_cfg = self.config["split"]

            working_series = series.copy()

            if trans_cfg.get("apply_log_transform", False):
                if(working_series <= 0).any():
                    logging.info("Log tranform requested but series has non-positive values - skipping")
                else:
                    working_series = np.log(working_series)
                    logging.info("Applied log tranform to series.")
            
            is_stationary, p_value = self.check_stationarity(working_series, label = "target series")

            test_size = split_cfg.get("test_size", 0.2)
            split_idx = int(len(working_series) * (1 - test_size))

            train_series = working_series.iloc[:split_idx]
            test_series = working_series.iloc[split_idx: ]

            logging.info(
                f"Chronological split complete. Train: {len(train_series)} rows, "
                f"Test: {len(test_series)} rows."
            )

            transform_info = {
                "log_transform_appiled": trans_cfg.get("apply_log_transform", False),
                "is_stationary": is_stationary,
                "adf_p_value": p_value,
            }

            artifacts_cfg = self.config["artifacts"]
            save_object(artifacts_cfg["transform_info_path"], transform_info)

            train_series.to_csv(artifacts_cfg["train_data_path"], header = True)
            test_series.to_csv(artifacts_cfg["test_data_path"], header = True)
            
            return train_series, test_series, transform_info

        except Exception as e:
            raise CustomException(e, sys)