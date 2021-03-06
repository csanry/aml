import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src import config, evaluation, plotting

warnings.filterwarnings("ignore")


def train(X_train, y_train, scorer, cv_split):

    # Setup the hyperparameter grid
    log_reg_param_grid = {
        # regularization param: higher C = less regularization
        "log_reg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        # specifies kernel type to be used
        "log_reg__penalty": ["l1", "l2", "none"],
    }

    # baseline model
    mm_scale = MinMaxScaler(feature_range=(0, 1))
    log_reg = LogisticRegression(
        solver="saga",
        penalty="none",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        max_iter=5000,
        warm_start=True,
    )

    # build the pipeline
    log_reg_pipe = Pipeline([("mm", mm_scale), ("log_reg", log_reg)])

    # Cross validate model with GridSearch
    log_reg_cv = GridSearchCV(
        estimator=log_reg_pipe,
        param_grid=log_reg_param_grid,
        scoring=scorer,
        refit="F_score",
        cv=cv_split,
        return_train_score=True,
        n_jobs=config.N_JOBS,
        verbose=10,
    )

    log_reg_cv.fit(X_train, y_train)

    log_reg_best_pipe = log_reg_cv.best_estimator_

    return log_reg_cv, log_reg_best_pipe


def evaluate(log_reg_cv, log_reg_best_pipe, X_test, y_test, file_name):

    log_reg_y_pred_prob = log_reg_best_pipe.predict_proba(X_test)[:, 1]
    log_reg_y_pred = log_reg_best_pipe.predict(X_test)

    evaluation.evaluate_tuning(tuner=log_reg_cv)

    report = evaluation.evaluate_report(
        y_test=y_test, y_pred=log_reg_y_pred, y_pred_prob=log_reg_y_pred_prob
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
        pickle.dump(log_reg_best_pipe, file)
