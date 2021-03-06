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
    rf_param_grid = {
        "rf__learning_rate": np.arange(0.8, 1.2, 0.05),
        "rf__subsample": np.arange(0.6, 0.9, 0.1),
        "rf__colsample_bynode": np.arange(0.6, 0.9, 0.1),
        "rf__max_depth": np.arange(3, 10, 1),
        "rf__n_estimators": np.arange(50, 200, 25),
        "rf__reg_alpha": list(np.linspace(0, 1)),
        "rf__reg_lambda": list(np.linspace(0, 1)),
    }

    # baseline model
    rf_clf = xgb.XGBRFClassifier(
        objective="binary:logistic",
        booster="gbtree",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    # build the pipeline
    rf_pipe = Pipeline([("rf", rf_clf)])

    # Cross validate model with RandomizedSearch
    rf_cv = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_grid,
        n_iter=30,
        scoring=scorer,
        refit="F_score",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
        random_state=config.RANDOM_STATE,
    )

    rf_cv.fit(X_train, y_train)

    rf_best_pipe = rf_cv.best_estimator_

    return rf_cv, rf_best_pipe


def evaluate(rf_cv, rf_best_pipe, X_test, y_test, file_name):

    evaluation.evaluate_tuning(tuner=rf_cv)

    rf_y_pred_prob = rf_best_pipe.predict_proba(X_test)[:, 1]
    rf_y_pred = rf_best_pipe.predict(X_test)

    report = evaluation.evaluate_report(
        y_test=y_test, y_pred=rf_y_pred, y_pred_prob=rf_y_pred_prob
    )

    plotting.plot_confusion_matrix(cf_matrix=report["cf_matrix"], model_name=file_name)
    plotting.plot_roc_curve(
        fpr=report["roc"][0],
        tpr=report["roc"][1],
        model_name=file_name,
        auc=report["auroc"],
    )

    save_path = config.MODEL_OUTPUT_PATH / f"{file_name}.pickle"

    with open(save_path, "wb") as file:
        pickle.dump(rf_best_pipe, file)

