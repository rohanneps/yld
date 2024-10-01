import pytest
from pandas import DataFrame
from yld_utils.classification import EvaluationMetricHelper
from yld_utils.constants import (
    ACCURACY_COL,
    F1_SCORE_COL,
    MODEL_COL,
    PRECISION_COL,
    RECALL_COL
)
from .TestBase import TestBase


class EvalMetricesTests(TestBase):
    @pytest.mark.data_eval
    def test_accuracy_from_data(self):
        acc_df: DataFrame = EvaluationMetricHelper.calculate_accuracy_from_dataframe(
            self.input_df
        )
        expected_acc_df: DataFrame = self._get_expected_output_df([ACCURACY_COL])
        assert acc_df.equals(expected_acc_df)

    @pytest.mark.data_eval
    def test_all_eval_metrices_from_data(self):
        eval_df: DataFrame = EvaluationMetricHelper.calculate_all_eval_metrices_from_data(
            self.input_df
        )
        expected_eval_df: DataFrame = self._get_expected_output_df([ACCURACY_COL, PRECISION_COL, RECALL_COL, F1_SCORE_COL])
        assert eval_df.equals(expected_eval_df)

    @pytest.mark.data_eval
    def test_f1_score_from_data(self):
        f1_score_df: DataFrame = (
            EvaluationMetricHelper.calculate_f1_score_from_dataframe(self.input_df)
        )
        expected_f1_score_df: DataFrame = self._get_expected_output_df([F1_SCORE_COL])
        assert f1_score_df.equals(expected_f1_score_df)

    @pytest.mark.data_eval
    def test_precision_and_recall_from_data(self):
        precision_recall_df: DataFrame = EvaluationMetricHelper.calculate_precision_and_recall_from_dataframe(
            self.input_df
        )
        expected_precision_recall_df: DataFrame = self._get_expected_output_df([PRECISION_COL, RECALL_COL])
        assert precision_recall_df.equals(expected_precision_recall_df)

    @pytest.mark.data_eval
    def test_precision_from_data(self):
        precision_df: DataFrame = (
            EvaluationMetricHelper.calculate_precision_from_dataframe(self.input_df)
        )
        expected_precision_df: DataFrame = self._get_expected_output_df([PRECISION_COL])
        assert precision_df.equals(expected_precision_df)

    @pytest.mark.data_eval
    def test_recall_from_data(self):
        recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_dataframe(
            self.input_df
        )
        expected_recall_df: DataFrame = self._get_expected_output_df([RECALL_COL])
        assert recall_df.equals(expected_recall_df)

    @pytest.mark.file_eval
    def test_accuracy_from_file(self):
        acc_df: DataFrame = EvaluationMetricHelper.calculate_accuracy_from_file(
            self.test_file_path
        )
        expected_acc_df: DataFrame = self._get_expected_output_df([ACCURACY_COL])
        assert acc_df.equals(expected_acc_df)

    @pytest.mark.file_eval
    def test_all_eval_metrices_from_file(self):
        eval_df: DataFrame = EvaluationMetricHelper.calculate_all_eval_metrices_from_file(
            self.test_file_path
        )
        expected_eval_df: DataFrame = self._get_expected_output_df([ACCURACY_COL, PRECISION_COL, RECALL_COL, F1_SCORE_COL])
        assert eval_df.equals(expected_eval_df)

    @pytest.mark.file_eval
    def test_f1_score_from_file(self):
        f1_score_df: DataFrame = EvaluationMetricHelper.calculate_f1_score_from_file(
            self.test_file_path
        )
        expected_f1_score_df: DataFrame = self._get_expected_output_df([F1_SCORE_COL])
        assert f1_score_df.equals(expected_f1_score_df)

    @pytest.mark.file_eval
    def test_precision_and_recall_from_file(self):
        precision_recall_df: DataFrame = EvaluationMetricHelper.calculate_precision_and_recall_from_file(
            self.test_file_path
        )
        expected_precision_recall_df: DataFrame = self._get_expected_output_df([PRECISION_COL, RECALL_COL])
        assert precision_recall_df.equals(expected_precision_recall_df)

    @pytest.mark.file_eval
    def test_precision_from_file(self):
        precision_df: DataFrame = EvaluationMetricHelper.calculate_precision_from_file(
            self.test_file_path
        )
        expected_precision_df: DataFrame = self._get_expected_output_df([PRECISION_COL])
        assert precision_df.equals(expected_precision_df)

    @pytest.mark.file_eval
    def test_recall_from_file(self):
        recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_file(
            self.test_file_path
        )
        expected_recall_df: DataFrame = self._get_expected_output_df([RECALL_COL])
        assert recall_df.equals(expected_recall_df)

    def _get_expected_output_df(self, col_names: list[str]) -> DataFrame:
        return DataFrame(data={MODEL_COL: self.model_list, **{col: [0.5]*2 for col in col_names}})
