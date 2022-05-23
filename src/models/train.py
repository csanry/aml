import logging
import pickle
import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src import config, evaluation, helpers
from src.models import adaboost, gbm, log_reg, rf, train_nca


def train_all():

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger()

    scorer = config.SCORER
    cv_split = config.CV_SPLIT

    X_train, X_test, y_train, y_test = helpers.read_files()

    model_options = [
        # adaboost,
        # gbm,
        log_reg,
    ]

    for model in model_options:
        logger.info(f"TRAINING {model.__name__}")
        cv, best_model = model.train(X_train, y_train, scorer, cv_split)

        logger.info(f"EVALUATION {model.__name__}")
        model.evaluate(X_train, X_test, y_train, y_test, cv, best_model)

        logger.info(f"DONE {model.__name__}")


train_all()
