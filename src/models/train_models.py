import logging

from src import config, helpers
from src.models import adaboost, gbm, log_reg, nca, rf, svm


def train_all():

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger()

    large_scorer, small_scorer = config.LARGE_SCORER, config.SMALL_SCORER
    cv_split = config.CV_SPLIT

    datasets = helpers.read_files()

    (
        X_large_train,
        y_large_train,
        X_large_test,
        y_large_test,
        X_small_train,
        y_small_train,
        X_small_test,
        y_small_test,
    ) = datasets.values()

    model_options = [
        adaboost,
        gbm,
        log_reg,
        rf,
        nca,
        svm,
    ]

    for key, data in zip(
        ["large", "small"],
        [
            [X_large_train, y_large_train, X_large_test, y_large_test],
            [X_small_train, y_small_train, X_small_test, y_small_test],
        ],
    ):
        scorer = large_scorer if key == "large" else small_scorer

        for model in model_options:
            logger.info(f"TRAINING {model.__name__}__{key}")
            cv, best_model = model.train(data[0], data[1], scorer, cv_split)

            logger.info(f"EVALUATION {model.__name__}__{key}")
            model.evaluate(
                data[2],
                data[3],
                cv,
                best_model,
                f"{model.__name__.split('.')[-1]}_{key}",
            )

            logger.info(f"DONE {model.__name__}__{key}")


if __name__ == "__main__":
    train_all()
