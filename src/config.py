# config.py
import os 

HOME_DIR = os.path.expanduser("~")

# epochs to train on
EPOCHS = 10

# path to model files
TRAIN_FILE_PATH  = os.path.join(HOME_DIR, "data", "final")
MODEL_OUTPUT = os.path.join(HOME_DIR, "models")

