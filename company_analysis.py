"""company_analysis.py
Works with edgartools==3.15.1
"""

from typing import List, Union, Literal

import numpy as np
import pandas as pd

from edgar import Company

from config.synonyms import ANNUAL_SYNONYMS
from config.synonyms import TTM_SYNONYMS
from config.synonyms import QUARTERLY_SYNONYMS

from xbrl_data_extraction import get_xbrl_statements, FinancialStatementExtractor
from calculation_observation_frequency import ObservationFrequencyDataCalculator, add_duration_labels
from config.financial_statement_config import AccountingItemKeys, FINANCIAL_STATEMENTS_CONFIG

ANNUAL = 'annual'
QUARTERLY = 'quarterly'
TTM = 'ttm'


class CompanyAnalysis:
    """
    Class definition of CompanyAnalysis
    * Load financials with caching (e.g. when you have downloaded latest 4 filings, no need to download again if you
    only need latest 2)
    *  Get annual, quarterly or trailing twelve month data for all three financial statements
    """

    def __init__(self, ticker: str):
        self._ticker = ticker.upper()
        company = Company(self.ticker)
        if company:
            self._company = company
        else:
            raise ValueError(f"The ticker '{ticker}' failed to be linked to a SEC registered corp.")
        self._filings = self.company.get_filings(form=['10-K', '10-Q'])  # This form filter does not work (see TSLA)
        self._filings_meta_data_df = self.filings.data.to_pandas()
        self._reporting_dates = self.filings_meta_data_df[self.filings_meta_data_df['form'].isin(['10-Q', '10-K'])]['reportDate'].tolist()
        self._data_extractor = FinancialStatementExtractor()

        self._cached_filings = None
        self._cached_xbrl_statements = None

    # XBRL statements and handle filings ---------------------------------------------------
    def _set_xbrl_statements(self, periods: int, annual_statements: bool) -> None:
        """
        Sets relevant XBRL statements for the data_extractor attribute for relevant filings based on the number of
        periods and if it is annual only or not
        :param periods: int
        :param annual_statements: bool
        :return: None
        """

        # Always assume that the XBRL statements are sorted by reporting date
        form = ['10-K'] if annual_statements else ['10-K', '10-Q']
        requested_filings = [f for f in self.filings if f.form in form][:periods]

        # Get a list of filings that has not yet been cached
        non_cached_filings = self._get_non_cached_filings(requested_filings=requested_filings)

        # If there are filings missing, load new XBRL statements and store them in cache
        if non_cached_filings:
            self._load_xbrl_statements(non_cached_filings=non_cached_filings)

        # Either retrieve annual statements or all of them
        if annual_statements:
            # Filter the 10-Ks
            xbrl_statements = self._get_cached_10k_xbrl_statements()
        else:
            xbrl_statements = list(self._cached_xbrl_statements.values()).copy()

        # Sort XBRL statement
        xbrl_statements = self._sort_xbrl_statements(xbrl_statements=xbrl_statements)

        self._data_extractor.xbrl_statements = xbrl_statements[:periods]
        return

    def _load_xbrl_statements(self, non_cached_filings: list) -> None:
        """
        Loads filings and corresponding XBRL statements and stores them in cache for future use
        :param non_cached_filings: list of filings
        :return: None
        """
        # If there are already some cached data, extend both the filings and XBRL statements that was already  loaded
        acc_no_list = self._get_accession_no_from_filings(filings=non_cached_filings)
        if self._cached_xbrl_statements:
            self._cached_filings.extend(non_cached_filings)
            self._cached_xbrl_statements.update(
                dict(zip(acc_no_list, get_xbrl_statements(filings=non_cached_filings)))
            )
        else:
            # Store filings and XBRL statements in cache
            self._cached_filings = non_cached_filings.copy()
            self._cached_xbrl_statements = dict(zip(acc_no_list, get_xbrl_statements(filings=non_cached_filings)))

    @staticmethod
    def _get_accession_no_from_filings(filings: list):
        return {f.accession_no for f in filings}

    @staticmethod
    def _sort_xbrl_statements(xbrl_statements: list) -> list:
        """
        Sorts the specified XBRL statements (earliest to oldest reporting date)
        :return: None
        """
        return sorted(xbrl_statements.copy(), key=lambda x: x.xbrls.entity_info['document_period_end_date'], reverse=True)

    def _get_non_cached_filings(self, requested_filings: list) -> list:
        """
        Returns a list of filings that has not yet been cached
        :param requested_filings: list of filings
        :return: list of filings
        """

        if not self._cached_filings:
            return requested_filings

        # Get accession numbers for comparison
        cached_acc_nos = {f.accession_no for f in self._cached_filings}
        non_cached_filings = [f for f in requested_filings
                              if f.accession_no not in cached_acc_nos]
        return non_cached_filings

    def _get_cached_10k_xbrl_statements(self) -> list:
        """
        Returns a list of stitched xbrl statements for 10-Ks from the list of cached statements
        :return: list
        """
        return [xbrl_stmt for xbrl_stmt in self._cached_xbrl_statements.values()
                if xbrl_stmt.xbrls.entity_info['document_type'] == '10-K']

    def clear_cache(self):
        """Clears all cached data"""
        self._cached_filings = None
        self._cached_xbrl_statements = None

    @staticmethod
    def _get_standardized_frequency(frequency: str) -> str:
        """
        Returns a standardized observation frequency str
        :param frequency: str
        :return: str
        """
        frequency_lower_no_blanks = frequency.lower().replace(' ', '').replace('_', '')
        if frequency_lower_no_blanks in ANNUAL_SYNONYMS:
            return ANNUAL
        elif frequency_lower_no_blanks in QUARTERLY_SYNONYMS:
            return QUARTERLY
        elif frequency_lower_no_blanks in TTM_SYNONYMS:
            return TTM
        else:
            raise ValueError(f"{frequency} is not a recognized observation frequency. Use either '{ANNUAL}', "
                             f"'{QUARTERLY}' or '{TTM}'")

    # Getting data and perform analysis ---------------------------------------------------
    def get_accounting_items(self, accounting_item: Union[str, List[str]], periods: int, frequency: str,
                             dates_as_cols: bool = True, dates_ascending_order: bool = True) -> pd.DataFrame:
        """
        Retrieve accounting items data for specified periods and frequency.

        Processes financial statement data to return accounting metrics in either:
        - Annual figures
        - Quarterly figures
        - Trailing Twelve Month (TTM) calculations

        Parameters
        ----------
        accounting_item : Union[str, List[str]]
            Accounting item(s) to retrieve. Can be a single item (str) or multiple items (List[str]).
            These should match the standardized accounting item names used in the XBRL statements.

        periods : int
            Number of historical periods to retrieve. For TTM calculations, additional periods
            are automatically fetched to ensure accurate rolling calculations.

        frequency : str
            The observation frequency for the returned data. Valid options are:
            - 'annual' - Annual financial statements
            - 'quarterly' - Quarterly financial statements
            - 'ttm' - Trailing Twelve Month calculations

        dates_as_cols : bool, optional
            If True (default), returns data with dates as columns (wide format).
            If False, returns data with dates in the index (long format).

        dates_ascending_order : bool, optional
            If True (default), returns data with dates in ascending order (oldest first).
            If False, returns dates in descending order (newest first).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the requested accounting items with the following characteristics:
            - Index: Accounting items (if dates_as_cols=True) or Dates (if dates_as_cols=False)
            - Columns: Dates (if dates_as_cols=True) or Accounting items (if dates_as_cols=False)
            - Values: The numerical accounting values for each item/period combination

        Notes
        -----
        - For TTM calculations, the method automatically fetches 3 additional periods of data
          to ensure accurate rolling calculations.
        - The method handles both instant (balance sheet) and duration (income/cash flow) items.
        - All returned values maintain their original units (e.g., millions) as reported in the filings.

        Examples
        --------
        >>> analyzer = CompanyAnalysis('AAPL')
        >>> # Get annual revenue for last 5 years
        >>> analyzer.get_accounting_items('revenues', 5, 'annual')
        >>> # Get quarterly Cost of Goods Sold for last 8 quarters (newest first)
        >>> analyzer.get_accounting_items('cogs', 8, 'quarterly', dates_ascending_order=False)
        >>> # Get TTM net income for last 4 periods
        >>> analyzer.get_accounting_items('net_income', 4, 'ttm')
        """
        frequency = self._get_standardized_frequency(frequency=frequency)
        # To be certain that we can calculate trailing twelve month use 3 more filings
        adj_periods = periods + 3 if TTM else periods

        # Extract the specified accounting items
        annual_statements_only = frequency == ANNUAL
        self._set_xbrl_statements(periods=adj_periods, annual_statements=annual_statements_only)
        extracted_acc_itm_df = self._data_extractor.get_accounting_items_dataframe(
            accounting_items=accounting_item,
        )

        # Add a duration suffix to period col name
        # Example: 'duration_2000-01-01_2000-12-31' -> 'duration_2000-01-01_2000-12-31__12mo'
        acc_itm_with_dur_cols_df = add_duration_labels(df=extracted_acc_itm_df)

        # Get the correct data columns based on the observation frequency (calculation is made when applicable)
        data_calculator = ObservationFrequencyDataCalculator(
            df=acc_itm_with_dur_cols_df,
            report_dates=self._reporting_dates
        )

        # Calculate quarterly values
        if frequency in [QUARTERLY, TTM]:

            qtr_acc_itm_df = data_calculator.get_quarterly_data(periods=adj_periods)

            if frequency == TTM:
                # Rolling across the columns by first transposing the result, sort index and then do the reverse
                # Only do this for data that is not instant like balance sheet items
                result_df = self._calculate_trailing_twelve_month(quarterly_df=qtr_acc_itm_df).iloc[:, :periods]  # Only include the requested periods
            else:
                # Result stays the same
                result_df = qtr_acc_itm_df.iloc[:, :periods]
        else:
            result_df = data_calculator.get_annual_data(periods=periods)

        # Format final result
        result_df = self._reformat_result_df(result_df=result_df, dates_as_cols=dates_as_cols,
                                             dates_ascending_order=dates_ascending_order)
        return result_df

    @staticmethod
    def _calculate_trailing_twelve_month(quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling sum of four quarters. Checks so that calculation is only performed on accouning items that
        are measured over a period (e.g. revenues) and not when measured in one point in time (e.g. total assets)
        :param quarterly_df: DataFrame(index=accounting item names, cols=data columns with dates as names)
        :return: DataFrame(index=accounting item names, cols=data columns with dates as names)
        """
        # First split the accounting items in instant (e.g. total assets) and periodic (e.g. revenues) data
        acc_item_type_map = {
            config.long_name: config.instant
            for config in FINANCIAL_STATEMENTS_CONFIG.values()
            if config.long_name in quarterly_df.index
        }
        # Boolean column that is True if instant else False
        quarterly_df['is_instant'] = quarterly_df.index.map(acc_item_type_map)

        ttm_acc_itm_df = quarterly_df[~quarterly_df['is_instant']].T.sort_index().rolling(window=4, min_periods=1).sum()
        ttm_acc_itm_df.iloc[:3, :] = np.nan  # The first 3 dates should not have a value
        ttm_acc_itm_df = ttm_acc_itm_df.sort_index(ascending=False).T  # Sort and transpose back to og format

        result = pd.concat([quarterly_df[quarterly_df['is_instant']], ttm_acc_itm_df], axis=0).reindex(quarterly_df.index)
        result.drop(columns=['is_instant'], inplace=True)
        return result

    @staticmethod
    def _reformat_result_df(result_df: pd.DataFrame, dates_as_cols: bool, dates_ascending_order: bool):
        """
        Returns a DataFrame that either sorts dates or sets dates as cols
        :param result_df: DataFrame(index=accounting items (str), cols=dates (str))
        :param dates_as_cols: bool (keeps dates as col headers else set accounting items as names)
        :param dates_ascending_order: bool if True sorts dates in ascending order (old to new)
        :return: DataFrame
        """
        df = result_df.copy()
        if dates_ascending_order:
            df.sort_index(axis=1, inplace=True)
        if dates_as_cols:
            df = df.T
        return df

    def _get_financial_statements(self, statement: Union[Literal['income_statement', 'balance_sheet', 'cashflow_statement'],
                                                    List[Literal['income_statement', 'balance_sheet', 'cashflow_statement']]], periods: int, frequency: str):
        raise NotImplemented('Not implemented yet...')

    def get_revenues(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.REVENUES, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_cost_of_goods_sold(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.COGS, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_cash_from_operations(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.CFO, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_depreciation_amortization_expense(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.DEPRECIATION_AMORTIZATION, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_sales_general_administration_expense(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.SALES_GENERAL_ADMIN, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_accounts_receivable(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.ACCOUNTS_RECEIVABLE, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_current_assets(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.CURRENT_ASSETS, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_current_liabilities(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.CURRENT_LIABILITIES, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_total_assets(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.TOTAL_ASSETS, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_property_plant_equipment(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.PPE, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_long_term_debt(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.LONG_TERM_DEBT, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    def get_net_income(self, periods: int, frequency: str, dates_as_cols: bool = True, dates_ascending_order: bool = True):
        return self.get_accounting_items(accounting_item=AccountingItemKeys.NET_INCOME, periods=periods, frequency=frequency,
                                         dates_as_cols=dates_as_cols, dates_ascending_order=dates_ascending_order)

    # Property getters/setters ---------------------------------------------------
    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, ticker: str):
        if ticker.upper() != self._ticker:
            self._ticker = ticker.upper()
            company = Company(ticker)
            if company:
                self._company = company
            else:
                raise ValueError(f"The ticker '{ticker}' failed to be linked to a SEC registered corp.")
            self._filings = self.company.get_filings(form=['10-K', '10-Q'])
            self._filings_meta_data_df = self.filings.data.to_pandas()
            self._filings_meta_data_df = self.filings.data.to_pandas()
            self._reporting_dates = self.filings_meta_data_df[self.filings_meta_data_df['form'].isin['10-K', '10-K']][
                'reportDate']
            self._data_extractor = FinancialStatementExtractor()
            self.clear_cache()

    @property
    def company(self):
        return self._company

    @property
    def filings(self):
        return self._filings

    @property
    def filings_meta_data_df(self):
        return self._filings_meta_data_df

    def __repr__(self):
        """Gives the precise representation so that the output can be recreated in code"""
        return f"CompanyAnalysis('{self.ticker}')"










