"""xbrl_data_extraction
Assumes edgartools==4.1.2
"""
import pandas as pd
import numpy as np
from typing import Literal, Union, List, Optional

from tools import get_col_names_with_yyyy_mm_dd
from accounting_item_extractor import AccountingItemExtractor
from financial_statement_config import FINANCIAL_STATEMENTS_CONFIG, items_with_multiple_statements, FinancialItemConfig
from enhanced_xbrls import EnhancedXBRLS

VALID_STATEMENT_LITERAL = Literal['income_statement', 'balance_sheet', 'cashflow_statement']
VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES = {
    'income_statement': 'IncomeStatement',
    'balance_sheet': 'BalanceSheet',
    'cashflow_statement': 'CashFlowStatement'
}


class FinancialStatementExtractor:

    def __init__(self, enhanced_xbrls: EnhancedXBRLS = None):
        self._enhanced_xbrls = enhanced_xbrls

    @staticmethod
    def _process_input(_input: {str, list, None}):
        """
        Returns a list of str after replacing blank with _ and lower case
        :param _input:
        :return:
        """

        if _input is None:
            return None

        elif isinstance(_input, str):
            _input = [_input]

        _input = [item.lower().replace(' ', '_') for item in _input.copy()]
        return _input

    def _process_statement_input(self, statement: Optional[Union[List[str], str]]) -> list:
        """
        Returns a list of statements that have been converted to lower case and blanks replaced by _ and check if they
        are valid
        :param statement: str or list of str
        :return: list of str
        """

        statement = self._process_input(_input=statement)

        if statement is None:
            return list(VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES.keys())

        if not all(isinstance(item, str) and item in VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES.keys() for item in statement):
            raise ValueError(f"All list items must be one of '%s'" % "', '".join(VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES.keys()))

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
            if acc_itm_config.statement == statement or any(stmt == statement for stmt in acc_itm_config.statement):
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
            raise ValueError("There is no config for some specified accounting item(s)")

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

        if self.enhanced_xbrls is None:
            raise ValueError("'enhanced_xbrls' has not been specified.")

        if (accounting_items is None) == (statement_accounting_item_configs_map is None):
            raise ValueError("Provide either accounting_items or statement_accounting_item_configs_map, not both")

        # If accounting_items was specified, create a map between statement and accounting items
        if accounting_items:
            statement_accounting_item_configs_map = self._get_accounting_item_configs(accounting_items=accounting_items)

        extractor = AccountingItemExtractor()  # Used to filter each DataFrame
        itm_multi_stmt = items_with_multiple_statements()
        mult_stmt_itm_loaded = dict.fromkeys(itm_multi_stmt, False)  # Keep track of items with multi statements

        result = []  # Initialize the result (will be a list of DataFrames that will be concatenated at the end)
        for statement, acc_itm_configs in statement_accounting_item_configs_map.items():

            dataframes = self.enhanced_xbrls.get_statement_dataframes(
                statement_type=VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES[statement]
            )
            extended_dataframes = None

            for config in acc_itm_configs:
                # Check if accounting item has already been loaded (happens if item exists in multiple statements)
                # If that is the case, continue to next item config in the list
                if mult_stmt_itm_loaded.get(config.long_name, False):
                    continue

                for idx, df in enumerate(dataframes):

                    extractor.df = df
                    filtered_by_total, extracted_df = self._extract_accounting_item(extractor=extractor, config=config)

                    if not filtered_by_total:
                        # it might not be extracted since concepts with custom prefixes like
                        # tsla:DepreciationAmortizationAndImpairment are not being properly handled at the moment
                        # (see 'Depreciation expense missing in Cash Flow statement (TSLA) #327' in edgartools github)

                        # Retrieve the extended DataFrame that includes more concepts but less meta data
                        if extended_dataframes is None:
                            extended_dataframes = self._get_extended_dataframes(statement=statement)

                        extended_df = extended_dataframes[idx]
                        extended_df = extended_df[~extended_df['label'].str.contains(' | ', regex=False, na=False)]
                        extractor.df = extended_df
                        if any('2024-01-01_2024-12-31' in col for col in extended_df.columns):
                            a = 3
                        _, extended_extracted_df = self._extract_accounting_item(extractor=extractor, config=config)

                        if not extracted_df.empty and not extended_extracted_df.empty:
                            sorted_columns = sorted(set(extracted_df.columns) | set(extended_extracted_df.columns), reverse=True)  # Column order

                            # Align columns using NumPy (faster than reindex for large DataFrames)
                            try:
                                values_combined = np.vstack([
                                    extracted_df[sorted_columns].values,
                                    extended_extracted_df[sorted_columns].values
                                ])
                                extracted_df = pd.DataFrame(values_combined, columns=sorted_columns,
                                                           index=extracted_df.index.union(extended_extracted_df.index))
                            except:
                                a = 4
                        elif not extended_extracted_df.empty:
                            extracted_df = extended_extracted_df.copy()

                    if not extracted_df.empty:
                        if extracted_df.shape[0] > 1:
                            # Sum the rows
                            extracted_df = self._sum_accounting_items_df(df=extracted_df)

                        # Extraction was made and found a non empty result so if this accounting item is included in
                        # more than one financial statement (e.g. net income)
                        if config.long_name in mult_stmt_itm_loaded:
                            mult_stmt_itm_loaded[config.long_name] = True
                    else:
                        # Add one row with nan (otherwise any requested accounting item that is missing will simply
                        # not be included in the final result)
                        extracted_df.loc[0] = np.nan

                    extracted_df.index = [config.long_name]

                    result.append(
                        extracted_df
                    )

        return self.combine_accounting_item_dataframes(dataframes=result)

    @staticmethod
    def _extract_accounting_item(extractor: AccountingItemExtractor, config: FinancialItemConfig) -> (bool, pd.DataFrame):
        """
        Returns a DataFrame where accounting items has been filtered according to a configuration
        :param extractor: AccountingItemExtractor
        :param config: FinancialItemConfig
        :return: DataFrame
        """
        extracted_df = extractor.extract_accounting_items(
            filter_config=config.filter_config.to_dict(),
            filter_by_total=config.filter_by_total
        )
        filtered_by_total = config.filter_by_total and extracted_df['is_total'].any()
        date_cols = get_col_names_with_yyyy_mm_dd(df=extracted_df)
        extracted_df = extracted_df[date_cols].copy()
        return filtered_by_total, extracted_df

    def _get_extended_dataframes(self, statement: str) -> List[pd.DataFrame]:
        """
        Returns a list of enhanced DataFrames
        :param statement: str
        :return: list of DataFrames
        """
        if statement == 'income_statement':
            return self.enhanced_xbrls.extended_income_statement_dataframes
        elif statement == 'balance_sheet':
            return self.enhanced_xbrls.extended_balance_sheet_dataframes
        elif statement == 'cashflow_statement':
            return self.enhanced_xbrls.extended_cashflow_statement_dataframes
        else:
            raise ValueError(f"'{statement}' is not a recognized statement")

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

    @property
    def enhanced_xbrls(self):
        return self._enhanced_xbrls

    @enhanced_xbrls.setter
    def enhanced_xbrls(self, enhanced_xbrls: EnhancedXBRLS):
        if enhanced_xbrls is not None and not isinstance(enhanced_xbrls, EnhancedXBRLS):
            raise ValueError(f"'enhanced_xbrls' needs to be of type {EnhancedXBRLS.__name__}")
        else:
            self._enhanced_xbrls = enhanced_xbrls
