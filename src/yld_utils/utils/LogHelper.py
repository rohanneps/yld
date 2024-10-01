import logging
import os
from typing import Any
from yld_utils.constants import LOG_DIR, LOG_FILE_NAME


class LogHelper:
    CONSOLE_LOG_LEVEL: int = logging.INFO
    LOG_FILE_NAME: str = os.path.join(LOG_DIR, LOG_FILE_NAME)
    LOG_FILE_LOG_LEVEL: int = logging.DEBUG

    def log_and_catch_exception(self, func) -> Any:
        def wrapper(*args, **kwargs):
            self._logger.debug(f"enter: {func.__name__}")
            output = None
            try:
                output = func(*args, **kwargs)
            except Exception as e:
                self._logger.error(e)
            self._logger.debug(f"exit: {func.__name__}")
            return output

        return wrapper

    def __init__(self) -> None:
        self._logger: logging.Logger
        self._init_logging()

    def _init_logging(self) -> None:
        """
        setup logging configuration
        """
        self._logger = logging.getLogger(__name__)
        self._logger.propagate = False
        self._logger.setLevel(self.LOG_FILE_LOG_LEVEL)

        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(self.LOG_FILE_NAME)
        c_handler.setLevel(self.CONSOLE_LOG_LEVEL)
        f_handler.setLevel(self.LOG_FILE_LOG_LEVEL)

        c_format = logging.Formatter("%(levelname)s: %(message)s")
        f_format = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        self._logger.addHandler(c_handler)
        self._logger.addHandler(f_handler)
