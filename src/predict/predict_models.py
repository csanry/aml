import logging

import pandas as pd
from src import config, helpers
from src.model_dispatcher import large_models, small_models


def predict_all(
    small_loans_model: str = "rf_300000", large_loans_model: str = "gbm_300000"
):

    logger = logging.getLogger()

    large_loans_data, small_loans_data = helpers.read_pred_files()

    large_model = large_models.get(large_loans_model)
    small_model = small_models.get(small_loans_model)

    for name, model, data in zip(
        [f"large_loans_{large_loans_model}", f"small_loans_{small_loans_model}"],
        [large_model, small_model],
        [large_loans_data, small_loans_data],
    ):
        logger.info(f"PREDICTING {name}")

        predictions = pd.Series(model.predict(data), name="prediction")
        final = pd.concat([data, predictions], axis=1)

        final.to_parquet(config.FIN_FILE_PATH / f"{name}_prediction.parquet")
        logger.info(
            f"DONE {name}; PREDICTIONS AT: {config.FIN_FILE_PATH / f'{name}_prediction.parquet'}"
        )

    logger.info(f"DONE")


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    predict_all(small_loans_model="rf_300000", large_loans_model="gbm_300000")

