import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import read_config, save_object, evaluate_arima_models, get_metrics


class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

    def initiate_model_training(self, train_series, test_series, train_exog=None, test_exog=None):
        """
        Grid searches ARIMA/SARIMAX over the p/d/q ranges in config.yaml.
        If train_exog/test_exog are provided (non-empty), fits SARIMAX with
        exogenous regressors. Picks the best model by test RMSE and saves it
        to artifacts/model.pkl.
        """
        logging.info("Starting model training")
        try:
            model_cfg = self.config["model"]
            trans_cfg = self.config["transformation"]
            artifacts_cfg = self.config["artifacts"]

            use_exog = train_exog is not None and not train_exog.empty

            best_order, best_seasonal_order, best_model, report = evaluate_arima_models(
                train_series=train_series,
                test_series=test_series,
                p_values=model_cfg["p_values"],
                d_values=model_cfg["d_values"],
                q_values=model_cfg["q_values"],
                seasonal=model_cfg.get("seasonal", False),
                seasonal_period=trans_cfg.get("seasonal_period", 7),
                train_exog=train_exog,
                test_exog=test_exog,
            )

            logging.info(f"Best order: {best_order}, seasonal_order: {best_seasonal_order}")

            if use_exog:
                forecast = best_model.forecast(steps=len(test_series), exog=test_exog)
            else:
                forecast = best_model.forecast(steps=len(test_series))

            metrics = get_metrics(test_series.values, forecast.values)

            logging.info(f"Best model metrics on test set: {metrics}")

            save_object(artifacts_cfg["model_path"], best_model)

            return {
                "best_order": best_order,
                "best_seasonal_order": best_seasonal_order,
                "metrics": metrics,
                "all_scores": report,
                "used_exog": use_exog,
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    series, exog_df = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_series, test_series, train_exog, test_exog, info = transformer.initiate_data_transformation(
        series, exog_df
    )

    trainer = ModelTrainer()
    results = trainer.initiate_model_training(train_series, test_series, train_exog, test_exog)

    print("Best order:", results["best_order"])
    print("Best seasonal order:", results["best_seasonal_order"])
    print("Used exog:", results["used_exog"])
    print("Metrics:", results["metrics"])
