README
==============================

1. [Project Organization](#1)
2. [Setting Up Environment](#2)
3. [Using a Docker Image](#3)
4. [Pipeline Workflow](#4)


Project Organization <a name="1"></a>
------------

    ├── LICENSE
    ├── Dockerfile
    ├── docker-compose.yml <- Docker files to set up a containerized environment
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for users of this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── final          <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---


Setting Up Environment <a name="2"></a>
------------

First, run the following terminal commands 

```
$ git clone https://github.com/csanry/aml.git
$ cd aml
```

Download and install [anaconda](https://www.anaconda.com/products/distribution)  

Then run the following to set up the conda environment and run the pipeline (requires linux distro - see steps below to 
setup docker image)

```
$ bash run_pipeline
```

You can check that the environment is correctly set up using the following command

```
$ make test_environment
```


Using a docker image <a name="3"></a>
------------

Download [docker](https://www.docker.com/products/docker-desktop/) and run the command 

```
$ docker-compose up
```

A docker container is created which the ubuntu linux distro and launches a Jupyter Lab environment for data science workflows

Run `docker-compose down` after you are done with your work

---


Pipeline Workflow <a name="4"></a>
------------

On running run_pipeline.sh, the following steps take place

* MAKING DATASET FROM RAW DATA: Checks if the dataset already exists locally, and if it does not, downloads it from cloud

* READING FROM THE LOCAL COPY: Reads the data downloaded locally in the previous step 

* INTERIM FILE PLACED IN INTERIM AND READY FOR FEATURE ENGINEERING: Data has been read and preprocessed and is ready for feature engineering

* FEATURE ENGINEERING: Feature engineering begins. Unwanted columns are dropped.

* BINNING NUMERICAL DATA: Numerical columns are assigned to categorical bins

* VECTORIZING CATEGORICAL DATA: Converting categorical variables to indicator variables

* IMPUTING MISSING DATA: Filling-in missing data using KNN Imputer

* TRAINING: Training models AdaBoost, GradientBoosting, Logistic Regression, RandomForestClassifier, KNeighborsClassifier and SupportVectorMachines. Each model is cross-validated using GridSearchCV and RandomizedSearchCV.

* EVALUATE: Evaluating each model post training. Performance evaluation metrics include: Accuracy, Precision, Recall, F1 Score, F2 Score, AUC and ROC curves. Best Model is that model that creates the right balance between low complexity and high sensitivity. 



