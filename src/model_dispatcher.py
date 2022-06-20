from src import config
from src.helpers import load_model

base_path = config.MODEL_OUTPUT_PATH

large_models = {
    "gbm": load_model(base_path / "gbm_large.pickle"),
    "log_reg": load_model(base_path / "log_reg_large.pickle"),
    "nca": load_model(base_path / "nca_large.pickle"),
    "rf": load_model(base_path / "rf_large.pickle"),
    "ada": load_model(base_path / "adaboost_large.pickle"),
    "svm": load_model(base_path / "svm_large.pickle")
}

small_models = {
    "gbm": load_model(base_path / "gbm_small.pickle"),
    "log_reg": load_model(base_path / "log_reg_small.pickle"),
    "nca": load_model(base_path / "nca_small.pickle"),
    "rf": load_model(base_path / "rf_small.pickle"),
    "ada": load_model(base_path / "adaboost_small.pickle"),
    "svm": load_model(base_path / "svm_small.pickle")
}