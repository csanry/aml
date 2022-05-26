import logging
import pickle
import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src import config, evaluation, helpers

warnings.filterwarnings("ignore")


def main():

    logger = logging.getLogger()

    scorer = config.SCORER
    cv_split = config.CV_SPLIT

    X_train, X_test, y_train, y_test = helpers.read_files()

    logger.info("HYPERPARAMETER TUNING")

    # baseline model
    adaboost_clf = AdaBoostClassifier(random_state=config.RANDOM_STATE)

    # Setup the hyperparameter grid
    adaboost_param_grid = {
        "n_estimators": np.arange(50, 300, 25),
    }

    # Cross validate model with RandomizedSearch
    adaboost_cv = GridSearchCV(
        estimator=adaboost_clf,
        param_grid=adaboost_param_grid,
        scoring=scorer,
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    # build the pipeline
    adaboost_pipe = Pipeline([("adaboost", adaboost_cv)])

    adaboost_pipe.fit(X_train, y_train)

    evaluation.evaluate_tuning(tuner=adaboost_cv)

    adaboost_best = adaboost_cv.best_estimator_

    # build the pipeline
    adaboost_best_pipe = Pipeline([("adaboost", adaboost_best)])

    adaboost_best_pipe.fit(X_train, y_train)

    adaboost_y_pred_prob = adaboost_best_pipe.predict_proba(X_test)[:, 1]
    adaboost_y_pred = adaboost_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=adaboost_y_pred, y_pred_prob=adaboost_y_pred_prob
    )

    filename = config.MODEL_OUTPUT_PATH / "adaboost.pickle"
    with open(filename, "wb") as file:
        pickle.dump(adaboost_best_pipe, file)

    logger.info("DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
