"""tools.py"""

import pandas as pd
import re
import numpy as np
from collections import defaultdict


def rename_duplicate_columns(df) -> pd.DataFrame:
    """
    Adds a suffix to any duplicate col headers
    :param df: DataFrame
    :return: DataFrame
    """

    df = df.copy()
    counts = defaultdict(int)
    new_columns = []

    for col in df.columns:
        counts[col] += 1
        new_columns.append(f"{col}.{counts[col]}" if counts[col] > 1 else col)

    df.columns = new_columns
    return df


def extract_date_str_column_headers(df):
    """
    Extracts column names that are in date format (YYYY-MM-DD).

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        list: List of column names that are date strings
    """
    date_columns = []
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # YYYY-MM-DD pattern

    for col in df.columns:
        if date_pattern.match(str(col)):
            date_columns.append(col)

    return date_columns


def find_matching_columns(
        source_df,
        target_df,
        source_column,
        return_first_match: bool = True,
        threshold: float = 1.0  # Default: 100% match
):
    """
    Find columns in target_df that have values matching source_column in source_df,
    based on a given threshold (e.g., 0.8 for 80% matching).

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame to compare against
        source_column: Column name in source_df to match
        return_first_match: If True, returns first match; else returns all matches
        threshold: Minimum fraction of matching values (0.0 to 1.0)

    Returns:
        list or str: Names of matching columns in target_df
    """
    matches = []
    source_series = source_df[source_column]

    for col in target_df.columns:
        target_series = target_df[col]

        # Check if lengths match (optional, but helps avoid incorrect comparisons)
        if len(source_series) != len(target_series):
            continue

        # Calculate the fraction of matching values
        match_fraction = (source_series == target_series).mean()

        if match_fraction >= threshold:
            if return_first_match:
                return col
            matches.append(col)

    return matches


def get_col_names_with_yyyy_mm_dd(df: pd.DataFrame) -> list:
    """
    Returns a list of all the col names that contains 'YYYY-MM-DD' in the col header for example
    "duration_2009-09-12_2003-05-12"
    :param df: DataFrame
    :return: list
    """

    pattern = r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)'  # Uses lookarounds instead of word boundaries
    result = []
    for col in df.columns:
        if re.findall(pattern, col):
            result.append(col)
    return result


def convert_if_numeric(s):
    """
    Convert only numeric-like columns for example '' becomes NaN
    Use df.apply(convert_if_numeric) to convert each column in the DataFrame
    :param s: series
    :return: series
    """
    try:
        return pd.to_numeric(s)
    except:
        return s


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
