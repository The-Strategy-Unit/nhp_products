from typing import List

import pandas as pd

from nhpy.az import load_data_file
from nhpy.utils import (
    get_logger,
)

logger = get_logger()


def suppress_small_counts(
    df: pd.DataFrame,
    suppress_cols: List[str],
    count_col: str = "baseline",
    threshold: int = 5,
) -> pd.DataFrame:
    """Suppression of small counts in detailed results. Filters dataframe to only rows with
    small numbers, then groups together values in specified columns and re-aggregates.

    Args:
        df (pd.DataFrame): Processed detailed results
        suppress_cols (List[str]): List of columns to use in suppressing small numbers,
        in order of preference.
        count_col (str, optional): The name of the column with the values to be suppressed
        and aggregated.
        Defaults to "value".
        threshold (int, optional): Maximum count allowed in the count_col

    Returns:
        pd.DataFrame: Processed detailed results with small counts aggregated together
    """
    logger.info("Beginning suppression...")
    full_index_cols = df.index.names.copy()
    for col in suppress_cols:
        df = df.reset_index()
        keep = df[df[count_col] >= threshold].copy()
        small = df[df[count_col] < threshold].copy()
        # avoid suppression if we don't need to
        if small.empty:
            df = df.groupby(full_index_cols).sum()
            break
        small[col] = "grouped"
        # reaggregate
        grouped = small.groupby(full_index_cols, dropna=False).sum().reset_index()
        # bind suppressed rows with the ones that didn't need suppression
        df = pd.concat([keep, grouped], ignore_index=True).groupby(full_index_cols).sum()
    return df.sort_index().sort_index(axis=1)


def aggregate_baseline():
    pass


def add_baseline_to_detailed_results():
    pass
