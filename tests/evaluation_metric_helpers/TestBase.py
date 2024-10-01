from unittest import TestCase
from pandas import DataFrame
from yld_utils.constants import CLASS_COL, MODEL_COL, PREDICTION_COL


class TestBase(TestCase):
    def setUp(self) -> None:
        self.model_list: list[str] = ["model1", "model2"]
        data: dict[str, list[str]] = {
            MODEL_COL: ["model1"] * 4 + ["model2"] * 4,
            CLASS_COL: ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            PREDICTION_COL: ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
        }
        self.input_df = DataFrame(data=data)
        self.test_file_path: str = "./test_model.csv"
