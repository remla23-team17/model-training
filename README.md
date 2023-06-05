# Model Training
The repository contains the training pipeline for the sentiment app for the CS4295 Project.

## 1. Training a Model
### Dependencies
Install all requirements with
```bash
pip install -r requirements.txt
```

### Dataset
Datasets are included in the data folder.

### Train
To train and evaluate a model on the provided data run the `main.py`. Individual stages on the training pipeline can be found in `/pipeline`

### Output
All output can be found in `/output`

## 2. DVC
### Setup
Setup DVC with your preferred storage setup. Run `dvc repro` to execute training pipeline.

### Experiments
To run an experiment and save the results run `dvc exp run`. To find the difference with current results use `dvc metrics diff` 


## 3. Pylint & DSLinter
Pylint and DSLinter configurations can be found in `.pylintrc`.

### Run pylint
```bash
pylint ./src
```

### Run DSLinter
```bash
pylint --load-plugins=dslinter ./src
```


## 4. Pytest Suite
Run tests from the root project folder

### Run all tests
```bash
pytest
```