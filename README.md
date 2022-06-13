CS610 - Applied Machine Learning - Loan Defaults
==============================

Table of Contents
------------

| S/NO | Section |
| --- | --- |
| 1. | [About this Project](#1) | 
| 2. | [Workflow](#2) | 
| 3. | [Project Organization](#3) | 
| 4. | [Setup Environment](#4) | 
| 5. | [Teardown Environment](#5) | 
| 6. | [Development Workflow](#6) | 
| 7. | [Pull Requests](#7) | 


About this Project <a name="1"></a>
------------

In this project, we explore several options to serve fast, reliable predictions on the probability of loan default. Ideally, the best performing solution can be used to automate internal decision-making or credit scoring processes. 

As loans are one of the most important products and revenue streams for banks, it is critical for banks to minimise the number of bad loans within its portfolio. In some extreme cases, the federal government might be forced to step in and bail out a failing bank by using taxpayer’s money. Therefore, it is important for banks to develop machine learning solutions that can better predict bad loans based on the profile of the customer and the nature of the loan.


Workflow <a name="2"></a>
------------

The project contains two main pipelines

### Train Pipeline
```mermaid
graph LR;    
   make_dataset --> build_features --> train_models --> evaluate_models
```

### Predict Pipeline
```mermaid
graph LR;    
   make_dataset --> build_features --> predict_models --> visualise_predictions
```


| Components | Description |
| --- | --- |
| `make_dataset`  | 1. Checks if the dataset exists in `data/raw`<br>2. Reads the file and performs pre-processing<br>3. Save outputs in `data/interim` for feature engineering  |
| `build_features` | 1. Prepares train and validation set<br>2. Drops unnecessary columns for training<br>3. Encodes categorical variables<br>4. Bin numerical variables<br>5. Imputing missing data for MAR variables<br>6. Save outputs in `data/final`  |
| `train_model` | 1. Trains all candidate models<br>2. Tune hyperparameters for each model<br>3. Save model weights in `models` folder |
| `evaluate_model`| 1. Evaluate models based on pre-defined metrics<br>2. Output charts to `reports/figures` |
| `predict_model`| 1. Loads pre-trained models from `models`<br>2. Output predictions in a `.csv` format |
| `visualize_predictions`| 1. Loads predictions from models<br>2. Output visualizations |

Project Organization <a name="3"></a>
------------

The repository is structured in the following hierarchy


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
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks
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
    | 
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io




Setting up the environment <a name="4"></a>
------------

### Prerequisties 

* Download and install [anaconda](https://www.anaconda.com/products/distribution) 

* Download [docker](https://www.docker.com/products/docker-desktop/) 

* Download [git](https://git-scm.com/downloads) 


Run the following terminal commands 

```
$ git clone https://github.com/csanry/aml.git
$ cd aml
```


Ensure that you are logged into docker hub. Then run the following command to set up the docker environment 

```
$ docker-compose up
```
 
The command launches an Ubuntu-based distro, and a Jupyter Lab environment for running the pipelines. Launch the Lab environment from the terminal by clicking on the generated URL

In the environment, run the following commands in an open terminal 

```
$ cd project
$ bash run_pipeline.sh
```

Check that the environment is correctly set up using the following command

```
$ make test_environment
```


Tearing down the environment <a name="5"></a>
------------

Close the browser by double tapping ctrl + c on the terminal

Run the following command on the terminal to tear down the environment 

```
docker-compose down
```


Development workflow <a name="6"></a>
------------

We will utilise the [github flow](https://githubflow.github.io/) philosophy where:

* Features should be developed on branches

* Whenever you think that the branch is ready for merging, open a [pull request](https://www.freecodecamp.org/news/how-to-make-your-first-pull-request-on-github-3/) 

* Why? Ensures that main branch is as clean and deployable as possible, no conflicts due to competing branches

* For more information, refer to this [article](https://githubflow.github.io/)

Submitting a pull request example <a name="7"></a>
------------

```bash
# checkout a branch
$ git checkout -b cs --track origin/main

# add and commit changes to the branch
$ git add .
$ git commit -m "message" -m "more detail on changes made" 

# push changes
$ git push origin cs
```

* Head to the main [repo](https://github.com/csanry/aml), find your branch, and click on "new pull request" 

* Enter a __descriptive__ title and description for your pull request

* Click on reviewers on the right side and request a review from `csanry`

* Select `create pull request` 

* For a visual explanation refer to [this document](/pr.pdf)




