"""
Configuration module for application-wide constants and custom exception types.

This module centralizes shared configuration elements including exit codes and
custom exception classes that provide detailed error context. It serves as a
single source of truth for error handling and status codes throughout the
application.
"""

import pandas as pd


# %%
# Exit codes
class ExitCodes:
    SUCCESS_CODE = 0
    ERROR_CODE = 1
    EXCEPTION_CODE = 2
    SIGINT_CODE = 130


# %%
# Colours
class Colours:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


# %%
# Constants
class Constants:
    PATH_DEPTH = 5
    TIMEOUT_SEC = 30

    # Age group boundaries
    AGE_UNKNOWN = -1
    AGE_INFANT = 0
    AGE_TODDLER = 5
    AGE_CHILD = 10
    AGE_ADOLESCENT = 16
    AGE_YOUNG_ADULT = 18
    AGE_ADULT = 35
    AGE_MIDDLE_AGE = 50
    AGE_SENIOR = 65
    AGE_ELDERLY = 75
    AGE_OLDEST = 85


# %%
# Custom error types


class EnvironmentVariableError(Exception):
    """Raised when required environment variables are missing."""

    def __init__(self, missing_vars: list[str], message: str | None = None):
        self.missing_vars = missing_vars
        self.message = (
            message
            or f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        super().__init__(self.message)


class EmptyContainerError(Exception):
    """Raised when an Azure Storage container exists but contains no blobs."""

    def __init__(self, container_name, path=None, message=None):
        self.container_name = container_name
        self.path = path
        self.message = (
            message
            or f"Container '{container_name}' {f'at path {path}' if path else ''} "
            "exists but contains no blobs"
        )
        super().__init__(self.message)


class DetailedResultsConfig:
    def __init__(
        self,
        ip_agg_cols: list[str],
        op_agg_cols: list[str],
        aae_agg_cols: list[str],
        custom_age_groups: bool,
    ):
        self.ip_agg_cols = ip_agg_cols
        self.op_agg_cols = op_agg_cols
        self.aae_agg_cols = aae_agg_cols
        self.custom_age_groups = custom_age_groups

    @staticmethod
    def age_groups(age: pd.Series) -> pd.Series:
        """Implemented in subclasses"""
        raise NotImplementedError()


class DetailedResultsStandard(DetailedResultsConfig):
    def __init__(self, custom_age_groups: bool = False):
        super().__init__(
            ip_agg_cols=[
                "sitetret",
                "age_group",
                "sex",
                "pod",
                "tretspef",
                "los_group",
                "maternity_delivery_in_spell",
            ],
            op_agg_cols=["sitetret", "pod", "age_group", "tretspef"],
            aae_agg_cols=[
                "sitetret",
                "pod",
                "age_group",
                "attendance_category",
                "aedepttype",
                "acuity",
            ],
            custom_age_groups=custom_age_groups,
        )


class DetailedResultsHRG(DetailedResultsConfig):
    def __init__(self, custom_age_groups: bool = True):
        super().__init__(
            ip_agg_cols=[
                "sitetret",
                "age_group",
                "sex",
                "pod",
                "tretspef",
                "sushrg",
                "maternity_delivery_in_spell",
            ],
            op_agg_cols=["sitetret", "pod", "age_group", "tretspef"],
            aae_agg_cols=[
                "sitetret",
                "pod",
                "age_group",
                "attendance_category",
                "aedepttype",
                "acuity",
            ],
            custom_age_groups=custom_age_groups,
        )

    @staticmethod
    def age_groups(age: pd.Series) -> pd.Series:
        """Cut age into groups

        Takes a pandas Series of age's and cut's into discrete intervals

        :param age: a Series of ages
        :type age: pandas.Series

        :returns: a Series of age groups
        :rtype: pandas.Series
        """
        return pd.cut(
            age.fillna(-1),
            [-1, 0, 1, 18, 1000],
            right=False,
            labels=[
                "Unknown",
                "0",
                "1-17",
                "18+",
            ],
        ).astype(str)
