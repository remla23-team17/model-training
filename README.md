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
Datasets are included in the data folder.

### Model Training
To train and evaluate a model on the provided data run the `main.py`. Individual stages on the training pipeline can be found in `/pipeline`

### Output
All output can be found in `/output`

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