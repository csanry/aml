import logging

import pandas as pd
import src.model_dispatcher
from src import config, helpers


def predict_all(small_loans_model: str = "rf", large_loans_model: str = "gbm"):

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger()

    large_loans_data, small_loans_data = helpers.read_pred_files()

    large_model = src.model_dispatcher.large_models.get(large_loans_model)
    small_model = src.model_dispatcher.small_models.get(small_loans_model)

    for name, model, data in zip(
        [f"large_loans_{large_loans_model}", f"small_loans_{small_loans_model}"],
        [large_model, small_model],
        [large_loans_data, small_loans_data],
    ):
        logger.info(f"PREDICTING {name}")

        predictions = pd.Series(model.predict(data), name="prediction")
        final = pd.concat([data, predictions], axis=1)

        logger.info(f"DONE {name}")
        final.to_parquet(config.FIN_FILE_PATH / f"{name}_prediction.parquet")


if __name__ == "__main__":
    predict_all()

