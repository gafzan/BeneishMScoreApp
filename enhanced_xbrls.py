"""enhanced_xbrl.py"""

import numpy as np
import pandas as pd
from edgar.xbrl import XBRLS, XBRL
from edgar.xbrl.stitching import StatementStitcher

from typing import List, Any, Literal, Dict


# This opts into pandas' future behavior, where downcasting will require explicit handling.
# The warning will disappear because you’ve explicitly enabled the new behavior.
# FutureWarning: The 'downcast' keyword in bfill is deprecated and will be removed in a future version.
# Use res.infer_objects(copy=False) to infer non-object dtype, or pd.to_numeric with the 'downcast' keyword to downcast
# numeric results.
# df[col_name] = df.filter(like=col_name).bfill(axis=1, downcast=False).iloc[:, 0]
pd.set_option('future.no_silent_downcasting', True)


class _EmptyPlaceholder:
    def __init__(self):
        self.mapping_store = None


# Overriding the function here since otherwise Streamlit will not be able to handle caching at an unknown location
def stitch_statements(
        xbrl_list: List[Any],
        statement_type: str = 'IncomeStatement',
        max_periods: int = 3,
) -> Dict[str, Any]:
    """
    Stitch together statements from multiple XBRL objects.

    Args:
        xbrl_list: List of XBRL objects, should be from the same company and ordered by date
        statement_type: Type of statement to stitch ('IncomeStatement', 'BalanceSheet', etc.)
        max_periods: Maximum number of periods to include (default: 3)

    Returns:
        Stitched statement data
    """
    print("Stitching statements...")
    # Initialize the stitcher
    stitcher = StatementStitcher(concept_mapper=_EmptyPlaceholder())

    # Collect statements of the specified type from each XBRL object
    statements = []

    for xbrl in xbrl_list:
        # Get statement data for the specified type
        statement = xbrl.get_statement_by_type(statement_type)
        if statement:
            statements.append(statement)

    # Stitch the statements
    return stitcher.stitch_statements(statements=statements, period_type=None, max_periods=max_periods, standard=False)


class EnhancedXBRL(XBRL):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass all args/kwargs to parent
        self._extended_income_statement = None
        self._extended_balance_sheet = None
        self._extended_cashflow_statement = None

    def _get_extended_statement_dataframe(self, statement_type: Literal['IncomeStatement', 'BalanceSheet', 'CashFlowStatement']):
        """
        Returns a DataFrame(cols='concept', 'label', value columns with period as col name, 'is_total')
        This DataFrame includes concepts that are custom like tsla:DepreciationAmortizationAndImpairment
        :param statement_type: str 'IncomeStatement', 'BalanceSheet', 'CashFlowStatement'
        :return: DataFrame
        """
        df = self.query().by_statement_type(statement_type).to_dataframe()

        # Only include rows that does not have concepts that starts with 'us-gaap'
        df = df[~df['concept'].str.startswith('us-gaap')].copy()

        if df.empty:
            return pd.DataFrame(
                columns=['concept', 'label', 'is_total']
            )

        # Add a period column
        df = self._add_period_column(df=df)

        # Add all the custom tags if any to the label
        df = self._add_info_to_label_col(df=df)

        # Drop some cols that are know to exist and are unnecessary
        needed_cols = ['concept', 'label', 'numeric_value', 'period']
        df = df[needed_cols].copy()
        df.drop_duplicates(keep='first', inplace=True)  # Some duplicates still remain
        # df.drop_duplicates(subset=df.columns.difference(['numeric_value']), keep='first', inplace=True)

        pivoted_df = (
            df.set_index(['concept', 'label', 'period'])['numeric_value']
            .unstack()
            .reset_index()
            .rename_axis(columns=None)
        )

        # Add a 'is_total' column with False
        pivoted_df['is_total'] = False
        return pivoted_df

    @staticmethod
    def _add_info_to_label_col(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom meta info columns to the label column separated by '|'
        :param df: DataFrame
        :return: DataFrame
        """

        # 1. Identify columns to the right of 'unit_ref'
        right_columns = df.columns[df.columns.get_loc('unit_ref') + 1:]

        # 2. Combine them into a single string (ignoring NaN values)
        df['label'] = df.apply(
            lambda row: row['label'] + ''.join(
                f" | {str(row[col])}" for col in right_columns
                if pd.notna(row[col])
            ),
            axis=1
        )

        # 3. Drop the original right columns if no longer needed
        df = df.drop(columns=right_columns)
        return df

    @staticmethod
    def _remove_duplicate_columns(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Drops duplicate columns and merges cols with `col_name` into one column,
        filling NaNs from the right.
        :param df: DataFrame
        :param col_name: str
        :return: DataFrame
        """
        df[col_name] = df.copy().filter(like=col_name).bfill(axis=1).values[:, 0]
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _add_period_column(df) -> pd.DataFrame:
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Initialize period column with empty strings
        df.insert(0, 'period', '')

        # Case 1: Both period_start and period_end exist
        if all(col in df.columns for col in ['period_start', 'period_end']):
            period_mask = df['period_start'].notna() & df['period_end'].notna()
            df.loc[period_mask, 'period'] = 'duration_' + df['period_start'] + '_' + df['period_end']

        # Case 2: period_instant exists
        if 'period_instant' in df.columns:
            instant_mask = df['period_instant'].notna()
            df.loc[instant_mask & (df['period'] == ''), 'period'] = 'instant_' + df['period_instant']

        # For rows where period is still empty (no period columns or all NaN), leave as empty string
        return df

    @property
    def extended_income_statement(self):
        if self._extended_income_statement is None:
            self._extended_income_statement = self._get_extended_statement_dataframe(statement_type='IncomeStatement')
        return self._extended_income_statement

    @property
    def extended_balance_sheet(self):
        if self._extended_balance_sheet is None:
            self._extended_balance_sheet = self._get_extended_statement_dataframe(statement_type='BalanceSheet')
        return self._extended_balance_sheet

    @property
    def extended_cashflow_statement(self):
        if self._extended_cashflow_statement is None:
            self._extended_cashflow_statement = self._get_extended_statement_dataframe(statement_type='CashFlowStatement')
        return self._extended_cashflow_statement


class EnhancedXBRLS(XBRLS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extended_income_statement_dataframes = None
        self._extended_balance_sheet_dataframes = None
        self._extended_cashflow_statement_dataframes = None

    @classmethod
    def from_filings(cls, filings: List[Any]) -> 'XBRLS':
        """
        Create an EnhancedXBRLS object from a list of Filing objects.

        Args:
            filings: List of Filing objects, should be from the same company

        Returns:
            EnhancedXBRLS object with stitched data
        """

        # Sort filings by date (newest first)
        sorted_filings = sorted(filings, key=lambda f: f.filing_date, reverse=True)

        # Create XBRL objects from filings
        enhanced_xbrl_list = []
        for filing in sorted_filings:
            try:
                enhanced_xbrl = EnhancedXBRL.from_filing(filing)
                enhanced_xbrl_list.append(enhanced_xbrl)
            except Exception as e:
                print(f"Warning: Could not parse XBRL from filing {filing.accession_number}: {e}")

        return cls(enhanced_xbrl_list)

    def get_statement_dataframes(self, statement_type: Literal['IncomeStatement', 'BalanceSheet', 'CashFlowStatement']) -> List[pd.DataFrame]:
        """
        Returns a list of financial statement DataFrames
        :param statement_type: str 'IncomeStatement', 'BalanceSheet', 'CashFlowStatement'
        :return: list of DataFrame
        """
        if statement_type not in ['IncomeStatement', 'BalanceSheet', 'CashFlowStatement']:
            raise ValueError("'statement_type' needs to be either of 'IncomeStatement', 'BalanceSheet', "
                             "'CashFlowStatement'")
        dataframes = [self._get_statement_dataframe(xbrl=xbrl, statement_type=statement_type)
                      for xbrl in self.xbrl_list]
        return dataframes

    @staticmethod
    def _get_statement_dataframe(xbrl, statement_type: Literal['IncomeStatement', 'BalanceSheet', 'CashFlowStatement']) -> pd.DataFrame:
        """
        Returns a DataFrame(columns=concept, label, data columns with duration, level, is_abstract, is_total, has_values)
        :param xbrl: Edgar XBRL
        :param statement_type: str
        :return: DataFrame
        """
        statement_df = pd.DataFrame(stitch_statements(xbrl_list=[xbrl], statement_type=statement_type,
                                                      max_periods=999,)['statement_data'])

        # Remove the 'decimals' column (who cares?)
        statement_df.drop(['decimals'], axis=1, inplace=True)

        # Only include the rows that has values
        statement_df = statement_df[statement_df['has_values']].drop(['has_values'], axis=1)

        # 'Normalize' the 'values' columns to create the duration columns and convert the numbers to numeric
        values_expanded = pd.json_normalize(statement_df['values'])  # Expands the dict in values column into separate cols

        # Merge the expanded values back with the original DataFrame (excluding the original 'values' column)
        statement_df = pd.concat(
            [statement_df.copy().drop(columns=['values']).reset_index(drop=True),
             values_expanded.reset_index(drop=True)],
            axis=1)

        if statement_type in ['CashFlowStatement']:
            # Drop rows containing 'instant' (Cash asset is sometimes included)
            statement_df.drop(columns=[col for col in statement_df.copy().columns if 'instant' in col], inplace=True)

        return statement_df

    @property
    def extended_income_statement_dataframes(self):
        if self._extended_income_statement_dataframes is None:
            self._extended_income_statement_dataframes = [xbrl.extended_income_statement for xbrl in self.xbrl_list]
        return self._extended_income_statement_dataframes

    @property
    def extended_balance_sheet_dataframes(self):
        if self._extended_balance_sheet_dataframes is None:
            self._extended_balance_sheet_dataframes = [xbrl.extended_balance_sheet for xbrl in
                                                          self.xbrl_list]
        return self._extended_balance_sheet_dataframes

    @property
    def extended_cashflow_statement_dataframes(self):
        if self._extended_cashflow_statement_dataframes is None:
            self._extended_cashflow_statement_dataframes = [xbrl.extended_cashflow_statement for xbrl in
                                                       self.xbrl_list]
        return self._extended_cashflow_statement_dataframes



