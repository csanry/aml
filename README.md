README
==============================

1. [Project Organization](#1)
2. [Downloading and using](#2)
3. [Using a docker image](#3)
4. [Development workflow](#4)
5. [Submitting a pull request](#5)


Project Organization <a name="1"></a>
------------

    ├── LICENSE
    ├── Dockerfile
    ├── docker-compose.yml <- Docker files to set up a containerized environment
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
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


Downloading and using <a name="2"></a>
------------

First, run the following terminal commands 

```
$ git clone https://github.com/csanry/aml.git
$ cd aml
```

Download and install [anaconda](https://www.anaconda.com/products/distribution) for your os 

Then run the following to create the ML environment

```
$ make create_environment
```

If you are using a Linux distro run 

```
$ make create_environment_linux
```

Check that the environment is correctly set up

```
$ make test_environment
```

---


Using a docker image <a name="3"></a>
------------

Download [docker](https://www.docker.com/products/docker-desktop/) and run the command 

```
$ docker-compose up
```

A docker container is created which the ubuntu linux distro and launchesj a jupyter lab environment for data science workflows

Run `docker-compose down` after you are done with your work

---


Development workflow <a name="4"></a>
------------

We will utilise the [github flow](https://githubflow.github.io/) philosophy where:

* Features should be developed on branches

* Whenever you think that the branch is ready for merging, open a [pull request](https://www.freecodecamp.org/news/how-to-make-your-first-pull-request-on-github-3/) 

* Why? Ensures that main branch is as clean and deployable as possible, no conflicts due to competing branches

* For more information, refer to this [article](https://githubflow.github.io/)

Submitting a pull request example <a name="5"></a>
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
