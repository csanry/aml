import logging
from typing import Dict

import pandas as pd
from src import config, helpers
from src.train import adaboost, gbm, log_reg, nca, rf, svm


def train_models(
    threshold: int,
    large_loans_file_paths: list,
    small_loans_file_paths: list,
    models_to_train: Dict[str, bool] = None,
):

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger()

    large_scorer, small_scorer = config.LARGE_SCORER, config.SMALL_SCORER
    cv_split = config.CV_SPLIT

    for large_loans_file in large_loans_file_paths:
        if large_loans_file.startswith("train"):
            ll_train = pd.read_parquet(config.FIN_FILE_PATH / large_loans_file)
            X_large_train = ll_train.drop(columns=config.TARGET)
            y_large_train = ll_train[config.TARGET]
        else:
            ll_test = pd.read_parquet(config.FIN_FILE_PATH / large_loans_file)
            X_large_test = ll_test.drop(columns=config.TARGET)
            y_large_test = ll_test[config.TARGET]

    for small_loans_file in small_loans_file_paths:
        if small_loans_file.startswith("train"):
            sl_train = pd.read_parquet(config.FIN_FILE_PATH / small_loans_file)
            X_small_train = sl_train.drop(columns=config.TARGET)
            y_small_train = sl_train[config.TARGET]
        else:
            sl_test = pd.read_parquet(config.FIN_FILE_PATH / small_loans_file)
            X_small_test = sl_test.drop(columns=config.TARGET)
            y_small_test = sl_test[config.TARGET]

    all_models = {
        "adaboost": adaboost,
        "gbm": gbm,
        "log_reg": log_reg,
        "rf": rf,
        "nca": nca,
        "svm": svm,
    }

    model_options = []
    if models_to_train:
        for model, to_train in models_to_train.items():
            if to_train:
                model_options.append(all_models.get(model))
    else:
        model_options = all_models.values()

    logger.info(f"TRAINING MODELS: {model_options}")

    for key, data in zip(
        ["large", "small"],
        [
            [X_large_train, y_large_train, X_large_test, y_large_test],
            [X_small_train, y_small_train, X_small_test, y_small_test],
        ],
    ):
        scorer = large_scorer if key == "large" else small_scorer

        for model in model_options:
            logger.info(f"TRAINING {model.__name__}_{threshold}_{key}")
            cv, best_model = model.train(
                X_train=data[0], y_train=data[1], scorer=scorer, cv_split=cv_split
            )

            logger.info(f"EVALUATION {model.__name__}_{threshold}_{key}")
            model.evaluate(
                cv,
                best_model,
                X_test=data[2],
                y_test=data[3],
                file_name=f"{model.__name__.split('.')[-1]}_{threshold}_{key}",
            )

            logger.info(f"DONE {model.__name__}_{threshold}_{key}")


if __name__ == "__main__":
    """ How to use argparse to specify which model(s) to train:
    Enter this command to find out: python3 src/models/train_models.py -h 
    Some examples:
    Running python3 src/models/train_models.py (without argparse) is equivalent to python3 src/models/train_models.py --models adaboost=true gbm=true log_reg=true rf=true nca=true svm=true
    python3 src/models/train_models.py -m aDAbOoST=true GBM=false log_reg=stringOtherThanTrueFalse --> {'adaboost': True, 'gbm': False, 'log_reg': False}
    """

    import argparse

    from src.helpers import restricted_int

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key.lower()] = (
                    True if value.lower() == "true" else False
                )

    parser = argparse.ArgumentParser(
        epilog="Eg: python3 src/models/train_models.py -m adaboost=true gbm=false --> adaboost will be trained but gbm will not"
    )

    parser.add_argument(
        "--threshold", type=restricted_int, default=300_000, required=True
    )

    parser.add_argument("--ll_files", nargs="*", default=[], required=False)

    parser.add_argument("--sl_files", nargs="*", default=[], required=False)

    parser.add_argument(
        "--models",
        nargs="*",
        action=ParseKwargs,
        help="specify whether to train a model. If argument is not given, all models will be trained.",
    )

    args = parser.parse_args()

    train_models(
        threshold=args.threshold,
        large_loans_file_paths=args.ll_files,
        small_loans_file_paths=args.sl_files,
        models_to_train=args.models,
    )
