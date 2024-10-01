
# Evaluation Metrics Util

This project is used to evaluate classification model performances


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
  EvaluationMetricHelper.calculate_accuracy_from_file("path/to/csv")
  EvaluationMetricHelper.calculate_f1_score_from_file("path/to/csv")
  EvaluationMetricHelper.calculate_precision_from_file("path/to/csv")
  EvaluationMetricHelper.calculate_recall_from_file("path/to/csv")

  EvaluationMetricHelper.calculate_precision_and_recall_from_file("path/to/csv")
  EvaluationMetricHelper.calculate_all_eval_metrices_from_file("path/to/csv")
```


## Authors

- [Rohan Amatya](https://github.com/rohanneps)

