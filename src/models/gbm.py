import pickle
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src import config, evaluation, plotting

warnings.filterwarnings("ignore")


def train(X_train, y_train, scorer, cv_split):

    # Setup the hyperparameter grid
    gbm_param_grid = {
        "gbm__learning_rate": np.arange(0.05, 0.4, 0.05),
        "gbm__max_depth": np.arange(3, 6, 1),
        "gbm__n_estimators": np.arange(50, 200, 25),
        "gbm__reg_alpha": list(np.linspace(0, 1)),
        "gbm__reg_lambda": list(np.linspace(0, 1)),
    }

    # baseline model
    gbm_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    # build the pipeline
    gbm_pipe = Pipeline([("gbm", gbm_clf)])

    # Cross validate model with RandomizedSearch
    gbm_cv = RandomizedSearchCV(
        estimator=gbm_pipe,
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

    gbm_cv.fit(X_train, y_train)

    gbm_best_pipe = gbm_cv.best_estimator_

    return gbm_cv, gbm_best_pipe


def evaluate(X_test, y_test, gbm_cv, gbm_best_pipe):
    gbm_y_pred_prob = gbm_best_pipe.predict_proba(X_test)[:, 1]
    gbm_y_pred = gbm_best_pipe.predict(X_test)

    evaluation.evaluate_tuning(tuner=gbm_cv)

    report = evaluation.evaluate_report(
        y_test=y_test, y_pred=gbm_y_pred, y_pred_prob=gbm_y_pred_prob
    )

    plotting.plot_confusion_matrix(report["cf_matrix"], "gbm")
    plotting.plot_roc_curve(report["roc"][0], report["roc"][1], "gbm", report["auroc"])

    filename = config.MODEL_OUTPUT_PATH / "gbm.pickle"
    with open(filename, "wb") as file:
        pickle.dump(gbm_best_pipe, file)
