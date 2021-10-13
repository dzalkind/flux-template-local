# ingest-template-local
Ingest template for a new pipeline that runs on your local filesystem.

## Prerequisites
Install Python 3.7+. We recommend using anaconda or venv to manage your python environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Hierarchy
* config
* data
* pipeline
* tests
* run_pipeline.py

## Running the pipeline from the command line
```bash
python3 run_pipeline.py [LIST_OF_FOLDERS_OR_FILES] --mode [prod OR dev]
```
Example:
```bash
python3 run_pipeline.py data/inputs --mode dev
```

Will process all the files in the data/inputs folder using the development configuration.

## Running/debugging the pipeline via a unit test

```bash
python3 tests/test_pipeline.py
```

Unit tests will run on all data placed in the data/inputs folder and will use
the development configuration.  We recommend using an IDE to run and debug your unit tests.