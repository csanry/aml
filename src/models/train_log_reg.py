import logging
import os
import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src import config, evaluation, helpers

warnings.filterwarnings("ignore")


def main():

    logger = logging.getLogger()

    scorer = config.SCORER
    cv_split = config.CV_SPLIT

    X_train, X_test, y_train, y_test = helpers.read_files()

    logger.info("HYPERPARAMETER TUNING")

    # baseline model
    log_reg = LogisticRegression(
        solver="saga",
        penalty="none",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        max_iter=5000,
        warm_start=True,
    )

    # Setup the hyperparameter grid
    log_reg_param_grid = {
        # regularization param: higher C = less regularization
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        # specifies kernel type to be used
        "penalty": ["l1", "l2", "none"],
    }

    # Cross validate model with GridSearch
    log_reg_cv = GridSearchCV(
        estimator=log_reg,
        param_grid=log_reg_param_grid,
        scoring=scorer,
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    mm_scale = MinMaxScaler(feature_range=(0, 1))

    # build the pipeline
    log_reg_pipe = Pipeline([("mm", mm_scale), ("log_reg", log_reg_cv)])

    log_reg_pipe.fit(X_train, y_train)

    evaluation.evaluate_tuning(tuner=log_reg_pipe[1])

    log_reg_best = log_reg_cv.best_estimator_

    # build the best pipeline
    log_reg_best_pipe = Pipeline([("mm", mm_scale), ("log_reg", log_reg_best)])

    log_reg_best_pipe.fit(X_train, y_train)

    log_reg_y_pred_prob = log_reg_best_pipe.predict_proba(X_test)[:, 1]
    log_reg_y_pred = log_reg_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=log_reg_y_pred, y_pred_prob=log_reg_y_pred_prob
    )

    filename = config.MODEL_OUTPUT_PATH / "log_reg.pickle"
    with open(filename, "wb") as file:
        pickle.dump(log_reg_best_pipe, file)

    logger.info("DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
