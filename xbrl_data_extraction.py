"""xbrl_data_extraction
Assumes edgartools==4.1.2
"""
import edgar.entity
import pandas as pd
import numpy as np
from typing import Literal, Union, List, Optional

from edgar.xbrl import XBRLS

from tools import get_col_names_with_yyyy_mm_dd

from accounting_item_extractor import AccountingItemExtractor
from financial_statement_config import FINANCIAL_STATEMENTS_CONFIG, items_with_multiple_statements


VALID_STATEMENT_LITERAL = Literal['income_statement', 'balance_sheet', 'cashflow_statement']
VALID_STATEMENTS = {'income_statement', 'balance_sheet', 'cashflow_statement'}


def get_xbrl_dataframe(xbrl_statement: edgar.xbrl.statements.Statements, statement_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame(columns=concept, label, data columns with duration, level, is_abstract, is_total, has_values)
    :param xbrl_statement: edgar.xbrl.statements.StitchedStatements
    :param statement_name: str
    :return: DataFrame
    """
    # DataFrames with concept, label, values as well as meta data like level, is_abstract, is_total, decimals and
    # has_values
    data = getattr(xbrl_statement, statement_name)(max_periods=999, standardize=False, use_optimal_periods=False)
    df = pd.DataFrame(data.statement_data['statement_data'])

    # Remove the 'decimals' column (who cares?)
    df.drop(['decimals'], axis=1, inplace=True)

    # Only include the rows that has values
    df = df[df['has_values']].drop(['has_values'], axis=1)

    # 'Normalize' the 'values' columns to create the duration columns and convert the numbers to numeric
    values_expanded = pd.json_normalize(df['values'])  # Expands the dict in values column into separate cols

    # Merge the expanded values back with the original DataFrame (excluding the original 'values' column)
    df = pd.concat([df.copy().drop(columns=['values']).reset_index(drop=True), values_expanded.reset_index(drop=True)], axis=1)

    if statement_name in ['cashflow_statement']:
        # Drop rows containing 'instant' (Cash asset is sometimes included)
        df.drop(columns=[col for col in df.copy().columns if 'instant' in col], inplace=True)

    return df


def get_xbrl_dataframes(xbrl_statements: list, statement_name: str) -> list:
    """
    Returns a list of DataFrames(columns=concept, label, data columns with duration, level, is_abstract, is_total,
    has_values)
    :param xbrl_statements:
    :param statement_name: str income_statement, balance_sheet, cashflow_statement
    :return: list of DataFrames
    """
    return [get_xbrl_dataframe(xbrl_statement=stmt, statement_name=statement_name) for stmt in xbrl_statements]


def get_xbrl_statements(filings: Union[edgar.entity.EntityFilings, edgar.entity.EntityFiling,
                        List[edgar.entity.EntityFiling]]) -> list:
    """
    Returns a list of XBRL statements extracted from specified filings
    :param filings: can either be one filing or filings or list of filing
    :return:
    """
    if isinstance(filings, edgar.entity.EntityFiling):
        filings = [filings]
    return [XBRLS.from_filings([filing]).statements for filing in filings if filing.form in ['10-K', '10-Q']]


class FinancialStatementExtractor:

    def __init__(self, xbrl_statements: Optional[list] = None):
        self.xbrl_statements = xbrl_statements

    @staticmethod
    def _process_input(_input):

        if _input is None:
            return None

        elif isinstance(_input, str):
            _input = [_input]

        _input = [item.lower().replace(' ', '_') for item in _input.copy()]
        return _input

    def _process_statement_input(self, statement):

        statement = self._process_input(_input=statement)

        if statement is None:
            return list(VALID_STATEMENTS)

        if not all(isinstance(item, str) and item in VALID_STATEMENTS for item in statement):
            raise ValueError(f"All list items must be one of {VALID_STATEMENTS}")

        return statement

    @staticmethod
    def _get_accounting_item_configs_for_statement(statement: str) -> list:
        """
        Find the accounting item config for each item in the specified financial statement
        :param statement: str e.g. 'income_statement'
        :return: list
        """

        result = []
        for acc_itm_config in FINANCIAL_STATEMENTS_CONFIG.values():
            if acc_itm_config['statement'] == statement or any(stmt == statement for stmt in acc_itm_config['statement']):
                result.append(acc_itm_config)
        return result

    def _get_accounting_item_configs(self, accounting_items: Union[List[str], str]) -> dict:
        """
        Returns a dict(keys=statement name, values=list of dict with the accounting item config)
        :param accounting_items: str or list of str (will be converted to lower case and blanks replaced by _ 
        :return: dict
        """""
        result = {}
        accounting_items = self._process_input(_input=accounting_items)
        if any(accounting_item not in FINANCIAL_STATEMENTS_CONFIG.keys() for accounting_item in accounting_items):
            raise ValueError("There is no config for some specified accounting item")
        # For each accounting item give, find the config dict and store it in a list
        for accounting_item in accounting_items:
            config = FINANCIAL_STATEMENTS_CONFIG[accounting_item]
            statements = config.statement
            # Concert to a list since some items have several statements (Net Income has cash flow and income statement)
            if isinstance(statements, str):
                statements = [statements]
            # Loop through each relevant statement and ad the config
            for statement in statements:
                if statement not in result:
                    # Add an empty list to initialize the result
                    result[statement] = []
                result[statement].append(config)
        return result

    def get_standard_financial_statement(self, statement: Optional[Union[VALID_STATEMENT_LITERAL, List[VALID_STATEMENT_LITERAL]]] = None) -> {dict, pd.DataFrame}:

        statements = self._process_statement_input(statement=statement)

        # Find the accounting item config for each item in the specified financial statement
        params = {
            stmt: self._get_accounting_item_configs_for_statement(statement=stmt)
            for stmt in statements
        }

        result = self.get_accounting_items_dataframe(statement_accounting_item_configs_map=params)

        if len(result) == 1:
            return result[list(result.keys())[0]]
        else:
            return result

    @staticmethod
    def _sum_accounting_items_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sums all rows per column. Ignores nan unless all rows are nan
        :param df: DataFrame
        :return: DataFrame
        """
        # Sum columns, ignoring NaN (but keeping 0s if they exist)
        col_sums_df = df.sum(axis=0, skipna=True)

        # Replace 0 with NaN ONLY if all values in the column were NaN
        mask_all_nan = df.isna().all(axis=0)  # True if entire column is NaN
        col_sums_df = col_sums_df.mask(mask_all_nan, np.nan)

        # Convert to DataFrame (single row, same columns as original)
        result_df = pd.DataFrame([col_sums_df], columns=df.columns)
        return result_df

    def get_accounting_items_dataframe(self, accounting_items: Optional[Union[List[str], str]] = None, statement_accounting_item_configs_map: Optional[dict] = None) -> pd.DataFrame:

        if self.xbrl_statements is None:
            raise ValueError("xbrl_statements has not been specified.")

        if (accounting_items is None) == (statement_accounting_item_configs_map is None):
            raise ValueError("Provide either accounting_items or statement_accounting_item_configs_map, not both")

        if accounting_items:
            statement_accounting_item_configs_map = self._get_accounting_item_configs(accounting_items=accounting_items)

        result = []  # Initialize the result (will be a list of DataFrames that will be concatenated at the end)
        extractor = AccountingItemExtractor()
        itm_multi_stmt = items_with_multiple_statements()
        mult_stmt_itm_loaded = dict.fromkeys(itm_multi_stmt, False)
        for statement, acc_itm_configs in statement_accounting_item_configs_map.items():

            dataframes = get_xbrl_dataframes(xbrl_statements=self.xbrl_statements, statement_name=statement)

            for df in dataframes:
                extractor.df = df

                for config in acc_itm_configs:
                    # Check if accounting item has already been loaded (happens if item exists in multiple statements)
                    # If that is the case, continue to next item config in the list
                    if mult_stmt_itm_loaded.get(config.long_name, False):
                        continue
                    extracted_df = extractor.extract_accounting_items(
                        filter_config=config.filter_config.to_dict(),
                        filter_by_total=config.filter_by_total
                    )
                    date_cols = get_col_names_with_yyyy_mm_dd(df=extracted_df)
                    extracted_df = extracted_df[date_cols].copy()
                    if not extracted_df.empty:
                        if extracted_df.shape[0] > 1:
                            # Sum the rows
                            extracted_df = self._sum_accounting_items_df(df=extracted_df)

                        # Extraction was made and found a non empty result so if this accounting item is included in
                        # more than one financial statement (e.g. net income)
                        if config.long_name in mult_stmt_itm_loaded:
                            mult_stmt_itm_loaded[config.long_name] = True
                    else:
                        # Add one row with nan (otherwise any requested accounting item that is missing will simply not
                        # be included in the final result)
                        extracted_df.loc[0] = np.nan

                    extracted_df.index = [config.long_name]

                    result.append(
                        extracted_df
                    )

        return self.combine_accounting_item_dataframes(dataframes=result)

    @staticmethod
    def combine_accounting_item_dataframes(dataframes: List[pd.DataFrame]):
        """
        Returns a DataFrame(index=accounting item names, col=data with duration str col name)
        First concatenates all DataFrames in the list and groups rows by their index (e.g. "Revenues") then takes the
        first non-NaN value for each column.
        :param dataframes: list of DataFrames
        :return: DataFrame
        """
        # Combine all DataFrames in the list
        combined_df = pd.concat(dataframes)

        # First group rows by their index (e.g. "Revenues" and "Cost of Goods Sold") then takes the first non-NaN value
        # for each column within each group
        df = combined_df.groupby(combined_df.index).first()
        return df
