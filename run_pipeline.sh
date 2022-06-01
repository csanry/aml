#!/bin/bash
# read -p "Enter environment name: " -r env_name
source /root/miniconda3/etc/profile.d/conda.sh 
conda activate aml
export PYTHONPATH="${PYTHONPATH}:/workspaces/aml"
# conda activate aml
python /workspaces/aml/src/data/make_dataset.py && python /workspaces/aml/src/features/build_features.py && python /workspaces/aml/src/models/train.py