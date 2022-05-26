from src import config
from src.helpers import load_model

base_path = config.MODEL_OUTPUT_PATH

models = {
    "gbm": load_model(base_path / "gbm.pickle"),
    "log_reg": load_model(base_path / "log_reg.pickle"),
    "nca": load_model(base_path / "nca.pickle"),
    "rf": load_model(base_path / "rf.pickle"),
    "ada": load_model(base_path / "adaboost.pickle"),
    "svm": load_model(base_path / "svm.pickle")
}
