import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config, save_object, evaluate_arima_models, get_metrics

class ModelTrainer:
    def __init__(self, config_path):
        self.config = read_config(config_path)

    def initiate_model_training(self, train_series, test_series):
        """
        Grid searches ARIMA (or SARIMAX if seasonal=True in config) over the
        p/d/q ranges in config.yaml, picks the best model by test RMSE,
        and saves it to artifacts/model.pkl.
        """

        logging.info("Starting model training")
        try:
            model_cfg = self.config["model"]
            trans_cfg = self.config["transformation"]
            artifacts_cfg = self.config["artifacts"]

            best_order, best_seasonal_order, best_model, report = evaluate_arima_models(
                train_series = train_series,
                test_series = test_series,
                p_values = model_cfg["p_values"], 
                d_values = model_cfg["d_values"],
                q_values = model_cfg["q_values"],
                seasonal = model_cfg.get("seasonal", False),
                seasonal_period = trans_cfg.get("seasonal_period", 7),
            )
            
            logging.info(f"Best ordder: {best_order}, seasonal_order: {best_seasonal_order}")

            forecast = best_model.forecast(steps = len(test_series))
            metrics = get_metrics(test_series.values, forecast.values)

            logging.info(f"Best model metrics on test set: {metrics}")

            save_object(artifacts_cfg["model_path"], best_model)

            return {
                "best_order": best_order,
                "best_seasonal_order": best_seasonal_order,
                "metrics": metrics,
                "all_scores": report,
            }
        
        except Exception as e:
            raise CustomException(e, sys)
