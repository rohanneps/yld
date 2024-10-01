
# Evaluation Metrics Util

This project is used to evalaute classification model performance.


## Tech Stack

Python, Pandas


## Run Locally

Clone the project

```bash
  git clone https://github.com/rohanneps/yld.git
```

Go to the project directory

Create and activate virtual environment

```bash
  python -m venv env
  source env/bin/activate
```

Install dependencies

```bash
  (env)pip install .
```

Run Test

```bash
  (env)pip install -e '.[testing]'
  (env)pytest tests 
```

Start Using

```python
  from yld_utils.classification import EvaluationMetricHelper
  precision_df = EvaluationMetricHelper.calculate_precision_from_dataframe("path/to/csv")
  print(precision_df)
```


## Authors

- [Rohan Amatya](https://github.com/rohanneps)

