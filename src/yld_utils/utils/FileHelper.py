import os
from typing import Optional

import pandas as pd
from pandas import DataFrame

from yld_utils.utils.LogHelper import LogHelper


class FileHelper:
    COLUMN_SEPARATOR: str = ","
    ENCODING: str = "utf-8"

    @classmethod
    def read_csv_from_path(
        cls, file_path: str, logger: LogHelper
    ) -> Optional[DataFrame]:
        if os.path.exists(file_path):
            return pd.read_csv(
                file_path, sep=cls.COLUMN_SEPARATOR, encoding=cls.ENCODING
            )
        logger._logger.debug(f"File {file_path} not found for evaluation")
