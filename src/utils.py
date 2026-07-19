import os
import sys
import yaml
import pickle
import itertools
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging


def read_config(config_path: str = "config/config.yaml") -> dict:
    """
    Reads the YAML config file and returns it as a dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """
    Saves any python object (model, dict, etc.) as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a pickled python object from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_arima_models(train_series, test_series, p_values, d_values, q_values,
                           seasonal=False, seasonal_period=7,
                           train_exog=None, test_exog=None):
    """
    Grid searches over (p, d, q) combinations (and seasonal order if enabled),
    fits ARIMA/SARIMAX on train_series (with exog if provided), forecasts
    len(test_series) steps, and scores each combination by test-set RMSE.

    If train_exog/test_exog are non-empty DataFrames, SARIMAX is used with
    exog regressors. Otherwise falls back to plain ARIMA with no exog.

    Returns:
        best_order, best_seasonal_order, best_model, report (dict of order -> rmse)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        use_exog = train_exog is not None and not train_exog.empty

        best_score, best_order, best_seasonal_order, best_model = float("inf"), None, None, None
        report = {}

        pdq_combinations = list(itertools.product(p_values, d_values, q_values))

        for order in pdq_combinations:
            try:
                if seasonal or use_exog:
                    # SARIMAX handles exog regressors; use a flat (non-seasonal)
                    # seasonal_order unless seasonal=True in config.
                    seasonal_order = (
                        (order[0], order[1], order[2], seasonal_period) if seasonal else (0, 0, 0, 0)
                    )
                    model = SARIMAX(
                        train_series,
                        exog=train_exog if use_exog else None,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    seasonal_order = None
                    model = ARIMA(train_series, order=order)

                fitted_model = model.fit()

                if use_exog:
                    forecast = fitted_model.forecast(steps=len(test_series), exog=test_exog)
                else:
                    forecast = fitted_model.forecast(steps=len(test_series))

                rmse = np.sqrt(mean_squared_error(test_series, forecast))
                report[str(order)] = rmse

                logging.info(f"Order {order} | seasonal_order {seasonal_order} -> RMSE: {rmse:.4f}")

                if rmse < best_score:
                    best_score = rmse
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = fitted_model

            except Exception as inner_e:
                # Some (p,d,q) combos fail to converge — skip them, don't kill the whole search
                logging.info(f"Order {order} failed to fit: {inner_e}")
                continue

        if best_model is None:
            raise Exception("No ARIMA/SARIMAX combination could be fit successfully on this data.")

        return best_order, best_seasonal_order, best_model, report

    except Exception as e:
        raise CustomException(e, sys)


def get_metrics(actual, predicted):
    """
    Returns a dict of common forecasting error metrics.
    """
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    except Exception as e:
        raise CustomException(e, sys)
