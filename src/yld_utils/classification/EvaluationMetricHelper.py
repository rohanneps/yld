from typing import Optional
from pandas import DataFrame
from yld_utils.utils.FileHelper import FileHelper
from yld_utils.utils.LogHelper import LogHelper
from yld_utils.constants import (
    ACCURACY_COL,
    CLASS_COL,
    F1_SCORE_COL,
    MODEL_COL,
    PREDICTION_COL,
    PRECISION_COL,
    RECALL_COL,
)

_logger = LogHelper()


class EvaluationMetricHelper:
    FALSE_POSTIVE_COL: str = "fp"
    FALSE_NEGATIVE_COL: str = "fn"
    TRUE_POSTIVE_COL: str = "tp"

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_accuracy_from_dataframe(cls, prediction_df: DataFrame) -> DataFrame:
        base_eval_metrics_df: DataFrame = cls._calculate_tp_fp_and_fn(prediction_df)
        model_total_preds_df = (
            prediction_df.groupby([MODEL_COL])[PREDICTION_COL].count().reset_index()
        )
        acc_df = model_total_preds_df.merge(
            base_eval_metrics_df[[MODEL_COL, cls.TRUE_POSTIVE_COL]]
            .groupby([MODEL_COL])
            .sum()
            .reset_index(),
            on=[MODEL_COL],
            how="inner",
        )
        acc_df[ACCURACY_COL] = acc_df[cls.TRUE_POSTIVE_COL] / acc_df[PREDICTION_COL]
        return cls._aggegrate_evaluation_by_model(ACCURACY_COL, acc_df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_accuracy_from_file(cls, file_path):
        df: Optional[DataFrame] = FileHelper.read_csv_from_path(file_path, _logger)
        if df is not None:
            return cls.calculate_accuracy_from_dataframe(df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_f1_score_from_dataframe(cls, prediction_df: DataFrame) -> DataFrame:
        precision_df: DataFrame = cls.calculate_precision_from_dataframe(prediction_df)
        recall_df: DataFrame = cls.calculate_recall_from_dataframe(prediction_df)
        base_eval_metrics_df = precision_df.merge(
            recall_df, on=[MODEL_COL], how="inner"
        )
        base_eval_metrics_df[F1_SCORE_COL] = (
            (2 * base_eval_metrics_df[PRECISION_COL] * base_eval_metrics_df[RECALL_COL])
        ) / (base_eval_metrics_df[PRECISION_COL] + base_eval_metrics_df[RECALL_COL])
        return cls._aggegrate_evaluation_by_model(F1_SCORE_COL, base_eval_metrics_df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_f1_score_from_file(cls, file_path):
        df: Optional[DataFrame] = FileHelper.read_csv_from_path(file_path, _logger)
        if df is not None:
            return cls.calculate_f1_score_from_dataframe(df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_precision_from_dataframe(cls, prediction_df: DataFrame) -> DataFrame:
        base_eval_metrics_df: DataFrame = cls._calculate_tp_fp_and_fn(prediction_df)
        base_eval_metrics_df[PRECISION_COL] = base_eval_metrics_df[
            cls.TRUE_POSTIVE_COL
        ] / (
            base_eval_metrics_df[cls.TRUE_POSTIVE_COL]
            + base_eval_metrics_df[cls.FALSE_POSTIVE_COL]
        )
        return cls._aggegrate_evaluation_by_model(PRECISION_COL, base_eval_metrics_df)

    @classmethod
    def _aggegrate_evaluation_by_model(cls, col_name: str, eval_metrics_df: DataFrame) -> DataFrame:
        return (
            eval_metrics_df[
                [
                    MODEL_COL,
                    col_name
                ]
            ]
            .groupby([MODEL_COL])
            .mean()
            .reset_index()
        )

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_precision_from_file(cls, file_path: str):
        df: Optional[DataFrame] = FileHelper.read_csv_from_path(file_path, _logger)
        if df is not None:
            return cls.calculate_precision_from_dataframe(df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_recall_from_dataframe(cls, prediction_df: DataFrame) -> DataFrame:
        base_eval_metrics_df: DataFrame = cls._calculate_tp_fp_and_fn(prediction_df)
        base_eval_metrics_df[RECALL_COL] = base_eval_metrics_df[
            cls.TRUE_POSTIVE_COL
        ] / (
            base_eval_metrics_df[cls.TRUE_POSTIVE_COL]
            + base_eval_metrics_df[cls.FALSE_NEGATIVE_COL]
        )
        return cls._aggegrate_evaluation_by_model(RECALL_COL, base_eval_metrics_df)

    @classmethod
    @_logger.log_and_catch_exception
    def calculate_recall_from_file(cls, file_path: str):
        df: DataFrame = FileHelper.read_csv_from_path(file_path, _logger)
        if df is not None:
            return cls.calculate_recall_from_dataframe(df)

    @classmethod
    @_logger.log_and_catch_exception
    def _calculate_tp_fp_and_fn(cls, prediction_df: DataFrame) -> DataFrame:
        """
        Return model-class-wise TP FP and FN
        """
        prediction_df[cls.TRUE_POSTIVE_COL] = (
            prediction_df[CLASS_COL] == prediction_df[PREDICTION_COL]
        )

        # compute TP per model-class
        tp_df = (
            prediction_df[prediction_df[cls.TRUE_POSTIVE_COL]]
            .groupby([MODEL_COL, CLASS_COL])[PREDICTION_COL]
            .count()
            .reset_index()
            .rename(columns={PREDICTION_COL: cls.TRUE_POSTIVE_COL})
        )

        rem_df = prediction_df[prediction_df[cls.TRUE_POSTIVE_COL] != True]
        # compute FN per model-class
        fn_df = (
            rem_df.groupby([MODEL_COL, CLASS_COL])[PREDICTION_COL]
            .count()
            .reset_index()
            .rename(columns={PREDICTION_COL: cls.FALSE_NEGATIVE_COL})
        )

        # compute FN per model-class
        fp_df = (
            rem_df.groupby([MODEL_COL, PREDICTION_COL])[CLASS_COL]
            .count()
            .reset_index()
            .rename(
                columns={CLASS_COL: cls.FALSE_POSTIVE_COL, PREDICTION_COL: CLASS_COL}
            )
        )

        # merging base eval metrices
        eval_metrics_df = tp_df.merge(fn_df, on=[MODEL_COL, CLASS_COL], how="left")
        eval_metrics_df = eval_metrics_df.merge(
            fp_df, on=[MODEL_COL, CLASS_COL], how="left"
        )
        eval_metrics_df.fillna(value=0, inplace=True)
        # aggregate base eval metrices per model-class
        return eval_metrics_df
