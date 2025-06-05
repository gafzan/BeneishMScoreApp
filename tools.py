"""tools.py"""

import pandas as pd
import numpy as np


def make_df_numeric(df: pd.DataFrame, columns_to_make_numeric: list = None) -> pd.DataFrame:
    """
    Converts relevant column values to numeric.
    :param df: DataFrame
    :param columns_to_make_numeric: list of column names to convert to numeric
    :return: DataFrame
    """

    df = df.copy()
    columns_to_make_numeric = columns_to_make_numeric or df.columns

    for col in columns_to_make_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def columnwise_rolling_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Returns a DataFrame with a columnwise rolling sum ignoring str columns

    Example
        df = context  2023-03-31  2023-06-30  2023-09-30  2023-12-31
        0   Revenue         100         210         NaN         500
        1  Expenses          50         NaN         NaN         250

    Returns
            context  2023-03-31  2023-06-30  2023-09-30  2023-12-31
        0   Revenue         NaN       310.0       210.0       500.0
        1  Expenses         NaN        50.0         NaN       250.0

    :param df: DataFrame
    :param window: int
    :return: DataFrame
    """
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns

    # Transpose numeric data for rolling operation and make sure values are numeric
    # (else I get {DataError}DataError('Cannot aggregate non-numeric type: object')
    transpose_df = make_df_numeric(df=df[numeric_cols].T)
    rolling_df = transpose_df.rolling(window=window, min_periods=1).sum().T

    # Matrix that is 1 if there are enough columns to calculate value, else nan
    indicator = np.where(
        np.isnan(
            transpose_df.fillna(0).rolling(window=window).sum().T.values
        ), np.nan, 1)
    rolling_df *= indicator

    # Combine results with the non-numeric columns
    result = pd.concat([df[df.columns.difference(numeric_cols)], rolling_df], axis=1)

    return result
