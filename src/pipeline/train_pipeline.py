import os
import sys

# Allow running this file directly (python src/pipeline/train_pipeline.py)
# as well as as a module (python -m src.pipeline.train_pipeline)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path

    def run(self):
        try:
            logging.info("======= TRAIN PIPELINE STARTED =======")

            ingestion = DataIngestion(self.config_path)
            series, exog_df = ingestion.initiate_data_ingestion()

            transformation = DataTransformation(self.config_path)
            train_series, test_series, train_exog, test_exog, transform_info = (
                transformation.initiate_data_transformation(series, exog_df)
            )

            trainer = ModelTrainer(self.config_path)
            results = trainer.initiate_model_training(train_series, test_series, train_exog, test_exog)

            logging.info("======= TRAIN PIPELINE COMPLETE =======")

            print("\n===== TRAINING COMPLETE =====")
            print(f"Best order (p,d,q):      {results['best_order']}")
            print(f"Best seasonal order:     {results['best_seasonal_order']}")
            print(f"Used exog features:      {results['used_exog']} {transform_info['exog_columns']}")
            print(f"Stationarity (ADF):      {transform_info['is_stationary']} (p={transform_info['adf_p_value']:.4f})")
            print(f"Log transform applied:   {transform_info['log_transform_applied']}")
            print("\nTest set metrics:")
            for k, v in results["metrics"].items():
                print(f"  {k}: {v:.4f}")
            print("\nModel saved to artifacts/model.pkl")

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
