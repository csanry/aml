#!/bin/bash

source activate aml

pip install -e .

python3 src/data/make_dataset.py &&\
python3 src/data/split_dataset.py --threshold 300000 &&\
python3 src/features/build_features.py &&\
python3 src/models/train_models.py --models adaboost=true gbm=true log_reg=true rf=true nca=true svm=true
