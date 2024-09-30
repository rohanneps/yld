from unittest import TestCase
from pandas import DataFrame
from yld_utils.classification.EvaluationMetricHelper import EvaluationMetricHelper


class FailureTests(TestCase):
    def setUp(self) -> None:
        self._non_existent_file: str = "NoneExistentFile.csv"

    def test_WHEN_file_does_not_exists_THEN_accurancy_returns_None(self):
        acc_df: DataFrame = EvaluationMetricHelper.calculate_accuracy_from_file(
            self._non_existent_file
        )
        assert acc_df is None

    def test_WHEN_file_does_not_exists_THEN_f1_score_returns_None(self):
        f1_score_df: DataFrame = EvaluationMetricHelper.calculate_f1_score_from_file(
            self._non_existent_file
        )
        assert f1_score_df is None

    def test_WHEN_file_does_not_exists_THEN_precision_returns_None(self):
        precision_df: DataFrame = EvaluationMetricHelper.calculate_precision_from_file(
            self._non_existent_file
        )
        assert precision_df is None

    def test_WHEN_file_does_not_exists_THEN_recall_returns_None(self):
        recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_file(
            self._non_existent_file
        )
        assert recall_df is None
