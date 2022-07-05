import logging
from typing import Dict
from src import config, helpers
from src.models import adaboost, gbm, log_reg, nca, rf, svm


def train(models_to_train: Dict[str, bool]):

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

    all_models = {
        "adaboost": adaboost,
        "gbm": gbm,
        "log_reg": log_reg,
        "rf": rf,
        "nca": nca,
        "svm": svm,
    }

    model_options = []
    if not models_to_train:
        model_options = all_models.values()
    else:   
        for model, to_train in models_to_train.items():
            if to_train:
                model_options.append(all_models[model])

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
    """ How to use argparse to specify which model(s) to train:
    Enter this command to find out: python3 src/models/train_models.py -h 
    Some examples:
    Running python3 src/models/train_models.py (without argparse) is equivalent to python3 src/models/train_models.py --models adaboost=true gbm=true log_reg=true rf=true nca=true svm=true
    python3 src/models/train_models.py -m aDAbOoST=true GBM=false log_reg=stringOtherThanTrueFalse --> {'adaboost': True, 'gbm': False, 'log_reg': False}
    """
    import argparse
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key.lower()] = True if value.lower() == 'true' else False

    parser = argparse.ArgumentParser(epilog="Eg: python3 src/models/train_models.py -m adaboost=true gbm=false --> adaboost will be trained but gbm will not")
    parser.add_argument('-m', '--models', nargs='*', action=ParseKwargs, help="specify whether to train a model. If argument is not given, all models will be trained.")
    args = parser.parse_args()

    train(args.models)
