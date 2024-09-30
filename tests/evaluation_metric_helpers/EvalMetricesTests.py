from unittest import TestCase
from pandas import DataFrame
from yld_utils.classification.EvaluationMetricHelper import EvaluationMetricHelper
from yld_utils.constants import ACCURACY_COL, CLASS_COL, F1_SCORE_COL, MODEL_COL, PREDICTION_COL, PRECISION_COL, RECALL_COL


class EvalMetricesTests(TestCase):
    def setUp(self) -> None:
        self.model_list: list[str] = ["M1", "M2"]
        data: dict[str, list[str]] = {
            MODEL_COL: ["M1"]*4 + ["M2"]*4,
            CLASS_COL: ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            PREDICTION_COL: ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"]
        }
        self.input_df = DataFrame(data=data)

    def test_accuracy_from_data(self):
        acc_df: DataFrame = EvaluationMetricHelper.calculate_accuracy_from_dataframe(self.input_df)
        expected_acc_df: DataFrame = self._get_expected_output_df(ACCURACY_COL)
        assert (acc_df.equals(expected_acc_df))

    def test_f1_score_from_data(self):
        f1_score_df: DataFrame = EvaluationMetricHelper.calculate_f1_score_from_dataframe(self.input_df)
        expected_f1_score_df: DataFrame = self._get_expected_output_df(F1_SCORE_COL)
        assert (f1_score_df.equals(expected_f1_score_df))

    def test_precision_from_data(self):
        precision_df: DataFrame = EvaluationMetricHelper.calculate_precision_from_dataframe(self.input_df)
        expected_precision_df: DataFrame = self._get_expected_output_df(PRECISION_COL)
        assert (precision_df.equals(expected_precision_df))

    def test_recall_from_data(self):
        recall_df: DataFrame = EvaluationMetricHelper.calculate_recall_from_dataframe(self.input_df)
        expected_recall_df: DataFrame = self._get_expected_output_df(RECALL_COL)
        assert (recall_df.equals(expected_recall_df))

    def test_precision_from_file(self):
        file_path: str = "./model_output.csv"
        EvaluationMetricHelper.calculate_precision_from_file(file_path)
        EvaluationMetricHelper.calculate_recall_from_file(file_path)
        EvaluationMetricHelper.calculate_accuracy_from_file(file_path)
        EvaluationMetricHelper.calculate_f1_score_from_file(file_path)

    def _get_expected_output_df(self, col_name: str) -> DataFrame:
        return DataFrame(data={MODEL_COL: self.model_list, col_name: [0.5, 0.5]})
