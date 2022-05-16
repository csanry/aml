import logging
import pickle
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
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
    gbm_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    # Setup the hyperparameter grid
    gbm_param_grid = {
        "learning_rate": np.arange(0.05, 0.4, 0.05),
        "max_depth": np.arange(3, 6, 1),
        "n_estimators": np.arange(50, 200, 25),
        "reg_alpha": list(np.linspace(0, 1)),
        "reg_lambda": list(np.linspace(0, 1)),
    }

    # Cross validate model with RandomizedSearch
    gbm_cv = RandomizedSearchCV(
        estimator=gbm_clf,
        param_distributions=gbm_param_grid,
        n_iter=30,
        scoring=scorer,
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
        random_state=config.RANDOM_STATE,
    )

    # build the pipeline
    gbm_pipe = Pipeline([("gbm", gbm_cv)])

    gbm_pipe.fit(X_train, y_train)

    evaluation.evaluate_tuning(tuner=gbm_cv)

    gbm_best = gbm_cv.best_estimator_

    # build the pipeline
    gbm_best_pipe = Pipeline([("gbm", gbm_best)])

    gbm_best_pipe.fit(X_train, y_train)

    gbm_y_pred_prob = gbm_best_pipe.predict_proba(X_test)[:, 1]
    gbm_y_pred = gbm_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=gbm_y_pred, y_pred_prob=gbm_y_pred_prob
    )

    filename = config.MODEL_OUTPUT_PATH / "gbm.pickle"
    with open(filename, "wb") as file:
        pickle.dump(gbm_best_pipe, file)

    logger.info("DONE")


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
