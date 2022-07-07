.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = aml
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) --file requirements.txt -c conda-forge -y
endif
		@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Setup packages
pkg: 
	pip install -e .


## Push updated docker
docker_push: 
	docker build -t csanry/aml:latest 
	docker login 
	docker push csanry/aml:latest


## Update docker 
docker_update:
	docker login 
	docker pull csanry/aml:latest
	docker-compose up 


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete


## Pre-commits 
commits: 
	pre-commit install
	pre-commit run --all-files


## Run pipe
training_pipe: 
	python3 src/data/make_dataset.py &&\
	python3 src/data/split_dataset.py --threshold 300_000 &&\
	python3 src/features/train_test_split_data.py \
	 --files df_small_loans_300000.parquet df_large_loans_300000.parquet &&\
	python3 src/features/feature_engineering.py \
	 --files train_init_df_large_loans_300000.parquet train_init_df_small_loans_300000.parquet \
	 test_init_df_large_loans_300000.parquet test_init_df_small_loans_300000.parquet &&\
	python3 src/train/train_models.py \
	 --threshold 300_000 \
	 --ll_files train_df_large_loans_300000.parquet test_df_large_loans_300000.parquet \
	 --sl_files train_df_small_loans_300000.parquet test_df_small_loans_300000.parquet \
	 --models adaboost=true gbm=true log_reg=true rf=true nca=true svm=true


train_thresholds:
	python3 src/data/make_dataset.py &&\
	python3 src/data/split_dataset.py --threshold 200_000 &&\
	python3 src/data/split_dataset.py --threshold 400_000 &&\
	python3 src/data/split_dataset.py --threshold 500_000

	python3 src/features/train_test_split_data.py \
	 --files \
	 df_small_loans_200000.parquet df_large_loans_200000.parquet \
	 df_small_loans_400000.parquet df_large_loans_400000.parquet \
	 df_small_loans_500000.parquet df_large_loans_500000.parquet &&\
	python3 src/features/feature_engineering.py \
	 --files \
	 train_init_df_large_loans_200000.parquet train_init_df_small_loans_200000.parquet \
	 test_init_df_large_loans_200000.parquet test_init_df_small_loans_200000.parquet \
	 train_init_df_large_loans_400000.parquet train_init_df_small_loans_400000.parquet \
	 test_init_df_large_loans_400000.parquet test_init_df_small_loans_400000.parquet \
	 train_init_df_large_loans_500000.parquet train_init_df_small_loans_500000.parquet \
	 test_init_df_large_loans_500000.parquet test_init_df_small_loans_500000.parquet

	python3 src/train/train_models.py --threshold 200_000 \
	 --ll_files train_df_large_loans_200000.parquet test_df_large_loans_200000.parquet \
	 --sl_files train_df_small_loans_200000.parquet test_df_small_loans_200000.parquet \
	 --models adaboost=false gbm=true log_reg=false rf=true nca=false svm=false
	 
	python3 src/train/train_models.py --threshold 400_000 \
	 --ll_files train_df_large_loans_400000.parquet test_df_large_loans_400000.parquet \
	 --sl_files train_df_small_loans_400000.parquet test_df_small_loans_400000.parquet \
	 --models adaboost=false gbm=true log_reg=false rf=true nca=false svm=false

	python3 src/train/train_models.py --threshold 500_000 \
	 --ll_files train_df_large_loans_500000.parquet test_df_large_loans_500000.parquet \
	 --sl_files train_df_small_loans_500000.parquet test_df_small_loans_500000.parquet \
	 --models adaboost=false gbm=true log_reg=false rf=true nca=false svm=false



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help


.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
