"""xbrl_data_extraction
Assumes edgartools==4.1.2
"""
import pandas as pd
import numpy as np
from typing import Literal, Union, List, Optional

from tools import get_col_names_with_yyyy_mm_dd
from accounting_item_extractor import AccountingItemExtractor
from financial_statement_config import FINANCIAL_STATEMENTS_CONFIG, items_with_multiple_statements, FinancialItemConfig, AccountingItemKeys
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

        self._cached_statement_dataframes_map = {}
        self._cached_statement_extended_dataframes_map = {}

    def _clear_cache(self):
        self._cached_statement_dataframes_map = {}
        self._cached_statement_extended_dataframes_map = {}

    def _get_dataframes(self, statement: str, merge_with_extended: bool):
        """

        :param statement:
        :param merge_with_extended:
        :return:
        """

        if statement not in self._cached_statement_dataframes_map:
            self._cached_statement_dataframes_map[statement] = self.enhanced_xbrls.get_statement_dataframes(
                statement_type=VALID_INTERNAL_EXTERNAL_STATEMENT_NAMES[statement]
            )
        if merge_with_extended:
            if statement not in self._cached_statement_extended_dataframes_map:
                extended_dataframes = self._get_extended_dataframes(
                    statement=statement
                )
                dataframes = self._cached_statement_dataframes_map[statement]
                self._cached_statement_extended_dataframes_map[statement] = [
                    pd.concat([dfs[0], dfs[1]], axis=0).reset_index(drop=True)
                    for dfs in list(zip(dataframes, extended_dataframes))]
            return self._cached_statement_extended_dataframes_map[statement]
        else:
            return self._cached_statement_dataframes_map[statement]






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

    def _get_accounting_item_configs(self, accounting_items: Union[List[str], str]) -> List[FinancialItemConfig]:
        """
        Returns a list of accounting item config
        :param accounting_items: str or list of str (will be converted to lower case and blanks replaced by _
        :return: list of FinancialItemConfig
        """
        accounting_items = self._process_input(_input=accounting_items)
        result = []
        for item in accounting_items:
            if item not in FINANCIAL_STATEMENTS_CONFIG.keys():
                raise ValueError(f"No configuration in FINANCIAL_STATEMENTS_CONFIG exists for {item}")
            else:
                result.append(FINANCIAL_STATEMENTS_CONFIG[item])
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

    def _extract_net_income(self, net_income_config: FinancialItemConfig, extractor: AccountingItemExtractor):

        # First do a not so strict filter on Net Income. If there are more than one result, find the Net Income per
        # share that closest lines up with the disclosed EPS figure

        net_income_df = self._extract_accounting_item(extractor=extractor, config=net_income_config)
        if net_income_df.shape[0] > 1:
            # Need to find the correct Net Income line to use
            net_income_df.reset_index(drop=True, inplace=True)
            eps_df = self._extract_accounting_item(extractor=extractor, config=FINANCIAL_STATEMENTS_CONFIG[AccountingItemKeys.EPS_BASIC])
            num_shares_df = self._extract_accounting_item(extractor=extractor, config=FINANCIAL_STATEMENTS_CONFIG[AccountingItemKeys.NUM_SHARES_BASIC])
            if eps_df.shape[0] == num_shares_df.shape[0] == 1:
                sum_abs_eps_diff = abs(net_income_df / num_shares_df.values - eps_df.values).sum(axis=1)
                net_income_df = net_income_df.loc[[sum_abs_eps_diff.sort_values().index[0]]].copy()  # Pick the row with the lowest difference between the published and calc. EPS
            else:
                net_income_df = net_income_df.iloc[[0], :]
        return net_income_df

    def get_accounting_items_dataframe(self, accounting_items: Optional[Union[List[str], str]]):

        if self.enhanced_xbrls is None:
            raise ValueError("'enhanced_xbrls' has not been specified.")

        extractor = AccountingItemExtractor()  # Used to filter each DataFrame
        extractor.add_prefixes_suffixes_to_ignore(ignore=f"{self.enhanced_xbrls.entity_info['ticker'].lower()}:")
        configs = self._get_accounting_item_configs(accounting_items=accounting_items)

        # Initialize both dict for storing extended and non-extended dataframes with caching
        result = []  # Initialize the result (will be a list of DataFrames that will be concatenated at the end)
        for config in configs:

            accounting_item_dataframes = self._get_accounting_item_dataframes(config=config, extractor=extractor)
            result.extend(accounting_item_dataframes)

        return self.combine_accounting_item_dataframes(dataframes=result)

    def _get_accounting_item_dataframes(self, config: FinancialItemConfig, extractor: AccountingItemExtractor) -> List[pd.DataFrame]:
        """
        Returns a list of DataFrames(index=accounting item long name, cols=period data
        :param config: FinancialItemConfig
        :param extractor: AccountingItemExtractor
        :return: list of DataFrame
        """

        # Get the non-extended data
        merge_with_extended = config.to_dict().get('use_extended_data', True)
        dataframes = self._get_dataframes(statement=config.statement,
                                          merge_with_extended=merge_with_extended)
        # For net income, since usually there are many similar line items to Net Income, use the EPS number (if it
        # exists) to find the closes match
        net_income = config.long_name == FINANCIAL_STATEMENTS_CONFIG[AccountingItemKeys.NET_INCOME].long_name
        result = []
        for df in dataframes:

            extractor.df = df
            if net_income:
                extracted_df = self._extract_net_income(net_income_config=config, extractor=extractor)
            else:
                extracted_df = self._extract_accounting_item(extractor=extractor,
                                                                config=config)

            if not extracted_df.empty:
                if extracted_df.shape[0] > 1:
                    # Sum the rows
                    extracted_df = self._sum_accounting_items_df(df=extracted_df)
            else:
                # Add one row with nan (otherwise any requested accounting item that is missing will simply
                # not be included in the final result)
                extracted_df.loc[0] = np.nan

            extracted_df.index = [config.long_name]

            result.append(
                extracted_df
            )
        return result

    def _extract_accounting_item(self, extractor: AccountingItemExtractor, config: FinancialItemConfig) -> pd.DataFrame:
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
        extracted_df = self._clean_duplicate_concepts(df=extracted_df)
        date_cols = get_col_names_with_yyyy_mm_dd(df=extracted_df)
        extracted_df = extracted_df[date_cols].copy()
        # Drop any rows where there are multiple rows with the same concept and contains ' | ' in label
        return extracted_df

    @staticmethod
    def _clean_duplicate_concepts(df: pd.DataFrame):
        """
        Remove rows where concepts are the same but labels contain ' | '.
        Keeps the first occurrence for each concept.

        Args:
            df (pd.DataFrame): Input DataFrame with 'label' and 'concept' columns

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Identify rows where label contains ' | '
        contains_pipe = df['label'].str.contains(' | ', regex=False, na=False)

        # Find concepts that appear in both regular and pipe-containing rows
        pipe_concepts = df[contains_pipe]['concept'].unique()
        regular_concepts = df[~contains_pipe]['concept'].unique()

        # Concepts that appear in both sets
        duplicate_concepts = set(pipe_concepts) & set(regular_concepts)

        # Get indices of rows to remove (pipe-containing rows for duplicate concepts)
        to_remove = df[contains_pipe & df['concept'].isin(duplicate_concepts)].index

        # Return DataFrame without these rows
        return df.drop(to_remove)

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

    def _get_extended_dataframes_new(self, statement: str) -> List[pd.DataFrame]:
        """
        Returns a list of enhanced DataFrames
        :param statement: str
        :return: list of DataFrames
        """
        if statement == 'income_statement':
            result = self.enhanced_xbrls.extended_income_statement_dataframes
        elif statement == 'balance_sheet':
            result = self.enhanced_xbrls.extended_balance_sheet_dataframes
        elif statement == 'cashflow_statement':
            result = self.enhanced_xbrls.extended_cashflow_statement_dataframes
        else:
            raise ValueError(f"'{statement}' is not a recognized statement")

        result = [self._clean_extended_dataframe(df=df) for df in result.copy()]
        return result

    @staticmethod
    def _clean_extended_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a new DataFrame that removes columns where concepts are duplicates while label includes ' | '
        (could be a segment for example and is removed to avoid double counting)
        :param df: DataFrame
        :return: DataFrame
        """
        # Step 1: Identify duplicate concepts (excluding the first occurrence)
        duplicate_concepts = df['concept'].duplicated(keep='first')

        # Step 2: Identify labels containing ' | '
        contains_pipe = df['label'].str.contains(' | ', regex=False, na=False)

        # Step 3: Filter rows to keep (either not a duplicate OR doesn't contain pipe)
        filtered_df = df[~(duplicate_concepts & contains_pipe)].copy().reset_index(drop=True)
        return filtered_df

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
