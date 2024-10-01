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
        cls, file_path: str, logger: LogHelper, required_headers: list[str]
    ) -> Optional[DataFrame]:
        if os.path.exists(file_path):
            df: pd.DataFrame = pd.read_csv(
                file_path, sep=cls.COLUMN_SEPARATOR, encoding=cls.ENCODING
            )
            if set(required_headers) == set(df.columns):
                return df
            logger._logger.debug(f"Missing input column {','.join(list(set(required_headers) - set(df.columns)))}")
            return 
        logger._logger.debug(f"File {file_path} not found for evaluation")
