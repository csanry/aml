import pickle
import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src import config, evaluation, plotting

warnings.filterwarnings("ignore")


def train(X_train, y_train, scorer, cv_split):

    # baseline model
    adaboost_clf = AdaBoostClassifier(random_state=config.RANDOM_STATE)

    # Setup the hyperparameter grid
    adaboost_param_grid = {
        "adaboost__n_estimators": np.arange(50, 300, 25),
    }

    # build the pipeline
    adaboost_pipe = Pipeline([("adaboost", adaboost_clf)])

    # Cross validate model with RandomizedSearch
    adaboost_cv = GridSearchCV(
        estimator=adaboost_pipe,
        param_grid=adaboost_param_grid,
        scoring=scorer,
        refit="F2",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    adaboost_cv.fit(X_train, y_train)

    adaboost_best_pipe = adaboost_cv.best_estimator_

    return adaboost_cv, adaboost_best_pipe


def evaluate(X_test, y_test, adaboost_cv, adaboost_best_pipe):
    evaluation.evaluate_tuning(tuner=adaboost_cv)
    adaboost_y_pred_prob = adaboost_best_pipe.predict_proba(X_test)[:, 1]
    adaboost_y_pred = adaboost_best_pipe.predict(X_test)

    evaluation.evaluate_report(
        y_test=y_test, y_pred=adaboost_y_pred, y_pred_prob=adaboost_y_pred_prob
    )

    fpr, tpr, thresholds = roc_curve(y_test, adaboost_y_pred_prob)
    plotting.plot_roc_curve(fpr, tpr, "Adaboost")

    filename = config.MODEL_OUTPUT_PATH / "adaboost.pickle"
    with open(filename, "wb") as file:
        pickle.dump(adaboost_best_pipe, file)
