[![Testing](https://github.com/remla23-team17/model-training/actions/workflows/push.yml/badge.svg?branch=main)](https://github.com/remla23-team17/model-training/actions/workflows/push.yml)

# Model Training Repository
The repository contains the training pipeline for the sentiment app for the CS4295 Project.


## Model Download Links:

- BoW.pkl: https://github.com/remla23-team17/model-training/releases/latest/download/bow.pkl
- Model: https://github.com/remla23-team17/model-training/releases/latest/download/model


## 1. Model Training Pipelione
### Install Dependencies
Install all requirements with
```bash
pip install -r requirements.txt
```

### Dataset Selection
Datasets are managed via DVC. See section about DVC for more info. Pull data from dvc with
```bash
dvc pull
```

### Model Training
To train and evaluate a model on the provided data run the following command

```bash
python3 main.py
```
Individual stages on the training pipeline can be found in `/pipeline`.

### Output
All output can be found in `/output`. The folder contains the bow.pkl and the trained model, as well as a performance.json file containing several performance metrics of the training stage.

### Running individual pipeline steps

Preprocessing step of the pipeline can be run from the project root folder with
```bash
 python3 src/preprocess.py "data/HistoricDump.tsv"
 python3 src/preprocess.py "data/FreshDump.tsv"
```

Training step of the pipeline can be run with
```bash
 python3 src/train.py "data/preprocessed_HistoricDump.tsv"
```

Testing step of the pipeline can be run with
```bash
  python3 src/test.py "data/preprocessed_FreshDump.tsv"
```

## 2. DVC
### Setup
Setup DVC with your preferred storage setup. Run the following command to execute training pipeline

### Retrieve data
Datasets are managed via DVC. See section about DVC for more info. Pull data from dvc with
```bash
dvc pull
```


### Test training pipeline
Run the following command to execute training pipeline
```bash
dvc repro
```

### Experiments
To run an experiment and save the results run 

```bash
dvc exp run
```

To find the difference with current results use
```bash
dvc metrics diff
```

## 3. Code Quality
Pylint and DSLinter configurations can be found in `.pylintrc`.

### Run pylint
```bash
pylint ./src
```

### Run DSLinter
```bash
pylint --load-plugins=dslinter ./src
```

### Run MLLint
```bash
mllint run
```

## 4. Pytest Suite
Run tests from the root project folder

### Run all tests
```bash
pytest
```
