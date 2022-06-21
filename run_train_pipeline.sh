#!/bin/bash

source activate aml

pip install -e .

python3 src/data/make_dataset.py &&\
python3 src/features/build_features.py &&\
python3 src/models/train_models.py

