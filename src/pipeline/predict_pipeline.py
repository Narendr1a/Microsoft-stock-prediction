import os
import sys
import numpy as np
import pandas as pd

# Allow running this file directly (python src/pipeline/predict_pipeline.py)
# as well as as a module (python -m src.pipeline.predict_pipeline)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config, load_object


class PredictPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

    def _build_future_exog(self, steps, future_index):
        """
        Builds exog values for the forecast horizon. Future Volume/Open/High/Low
        are just as unknown as the target itself, so this is a simplifying
        assumption, NOT a real prediction of future exog:

        strategy "last_value" (default): carries the last observed exog row
        forward for every future step. Good enough for short horizons where
        exog doesn't swing much; gets less reliable the further out you forecast.
        """
        artifacts_cfg = self.config["artifacts"]
        future_cfg = self.config.get("future_exog", {})
        strategy = future_cfg.get("strategy", "last_value")

        test_exog_path = artifacts_cfg["test_exog_path"]
        train_exog_path = artifacts_cfg["train_exog_path"]

        if os.path.exists(test_exog_path):
            exog_history = pd.read_csv(test_exog_path, index_col=0, parse_dates=True)
        elif os.path.exists(train_exog_path):
            exog_history = pd.read_csv(train_exog_path, index_col=0, parse_dates=True)
        else:
            return None  # no exog was used during training

        if exog_history.empty:
            return None

        if strategy == "last_value":
            last_row = exog_history.iloc[[-1]]
            future_exog = pd.concat([last_row] * steps, ignore_index=True)
            future_exog.index = future_index
            logging.info(
                "Future exog built via last_value carry-forward — an assumption, "
                "not a real prediction of future Volume/Open/High/Low."
            )
            return future_exog
        else:
            raise ValueError(f"Unknown future_exog strategy: '{strategy}'")

    def forecast(self, steps=None):
        """
        Loads the saved model and forecasts `steps` steps into the future.
        If `steps` is None, uses model.forecast_horizon from config.yaml.
        Automatically reverses the log transform if it was applied during training.
        Returns a pandas Series indexed by forecasted dates.
        """
        try:
            artifacts_cfg = self.config["artifacts"]
            model_cfg = self.config["model"]

            if steps is None:
                steps = model_cfg.get("forecast_horizon", 30)

            model = load_object(artifacts_cfg["model_path"])
            transform_info = load_object(artifacts_cfg["transform_info_path"])

            # Build a future date index continuing from the training data's last date
            train_df = pd.read_csv(artifacts_cfg["train_data_path"], index_col=0, parse_dates=True)
            test_df = pd.read_csv(artifacts_cfg["test_data_path"], index_col=0, parse_dates=True)
            last_date = test_df.index.max() if len(test_df) else train_df.index.max()

            inferred_freq = pd.infer_freq(train_df.index) or "D"
            future_index = pd.date_range(
                start=last_date, periods=steps + 1, freq=inferred_freq
            )[1:]

            future_exog = None
            if transform_info.get("exog_columns"):
                future_exog = self._build_future_exog(steps, future_index)

            if future_exog is not None:
                forecast_values = model.forecast(steps=steps, exog=future_exog)
            else:
                forecast_values = model.forecast(steps=steps)

            if transform_info.get("log_transform_applied", False):
                forecast_values = np.exp(forecast_values)
                logging.info("Reversed log transform on forecast output.")

            forecast_series = pd.Series(forecast_values.values, index=future_index, name="forecast")

            logging.info(f"Generated forecast for {steps} steps ahead.")
            return forecast_series

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictPipeline()
    result = predictor.forecast()
    print("\n===== FORECAST =====")
    print(result)
    if not result.empty:
        exog_cols = read_config().get("data", {}).get("exog_columns", [])
        if exog_cols:
            print(
                f"\nNote: exog features {exog_cols} for these future dates were "
                f"estimated via last-value carry-forward, not actually known — "
                f"see future_exog.strategy in config.yaml."
            )
