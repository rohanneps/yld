from unittest.mock import patch
import pytest
import pandas
from pandas import DataFrame
from yld_utils.classification import EvaluationMetricHelper
from yld_utils.constants import MODEL_COL
from yld_utils.utils import FileHelper
from .TestBase import TestBase


class FailureTests(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.test_file_path: str = "NoneExistentFile.csv"

    @pytest.mark.file_eval
    def test_WHEN_file_does_not_exists_THEN_accurancy_returns_None(self):
        acc_df: DataFrame = EvaluationMetricHelper.calculate_accuracy_from_file(self.test_file_path)
        assert acc_df is None

    @pytest.mark.file_eval
    def test_WHEN_file_does_not_exists_THEN_f1_score_returns_None(self):
        f1_score_df: DataFrame = EvaluationMetricHelper.calculate_f1_score_from_file(self.test_file_path)
        assert f1_score_df is None

    @pytest.mark.file_eval
    def test_WHEN_file_does_not_exists_THEN_precision_returns_None(self):
        precision_df: DataFrame = EvaluationMetricHelper.calculate_precision_from_file(self.test_file_path)
        assert precision_df is None

    @pytest.mark.file_eval
    def test_WHEN_file_does_not_exists_THEN_recall_returns_None(self):
        recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_file(self.test_file_path)
        assert recall_df is None

    @pytest.mark.file_eval
    def test_WHEN_input_file_has_missing_header_THEN_recall_returns_None(self):
        self.test_file_path: str = "./test_model.csv"
        def _get_incorrect_columns() -> DataFrame:
            df: DataFrame = pandas.read_csv(self.test_file_path, sep=FileHelper.COLUMN_SEPARATOR)
            df.rename(columns={MODEL_COL: "new_column"}, inplace=True)
            return df

        with patch.object(pandas, "read_csv", side_effect=[_get_incorrect_columns()]):
            recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_file(self.test_file_path)
        assert recall_df is None
