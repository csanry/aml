from src import config
from src.helpers import load_model

base_path = config.MODEL_OUTPUT_PATH

models = {
    "gbm_small": load_model(base_path / "gbm_small.pickle"),
    "gbm_large": load_model(base_path / "gbm_large.pickle"),
    "log_reg_small": load_model(base_path / "log_reg_small.pickle"),
    "log_reg_large": load_model(base_path / "log_reg_large.pickle"),
    "nca_small": load_model(base_path / "nca_small.pickle"),
    "nca_large": load_model(base_path / "nca_large.pickle"),
    "rf_small": load_model(base_path / "rf_small.pickle"),
    "rf_large": load_model(base_path / "rf_large.pickle"),
    "ada_small": load_model(base_path / "adaboost_small.pickle"),
    "ada_large": load_model(base_path / "adaboost_large.pickle"),
    "svm_small": load_model(base_path / "svm_small.pickle"),
    "svm_large": load_model(base_path / "svm_large.pickle")
}
