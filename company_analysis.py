"""company_analysis.py"""
import time
from typing import List, Optional, Dict
import pandas as pd
from datetime import date
from datetime import timedelta
import dateutil.parser

from edgar import set_identity
from edgar import Company, MultiFinancials

from synonyms import INCOME_STATEMENT_SYNONYMS
from synonyms import BALANCE_SHEET_SYNONYMS
from synonyms import CASH_FLOW_STATEMENT_SYNONYMS
from synonyms import ANNUAL_SYNONYMS
from synonyms import TTM_SYNONYMS
from synonyms import QUARTERLY_SYNONYMS

from tools import make_df_numeric
from tools import columnwise_rolling_sum

set_identity("gafzan@gmail.com")


class CompanyAnalysis:
    """
    Class definition of CompanyAnalysis
    * Load financials with caching (e.g. when you have downloaded latest 4 filings, no need to download again if you
    only need latest 2)
    *  Get annual, quarterly or trailing twelve month data for all three financial statements
    """

    def __init__(self, ticker: str):
        self._ticker = ticker.upper()
        company = Company(ticker)
        if company:
            self._company = company
        else:
            raise ValueError(f"The ticker '{ticker}' failed to be linked to a SEC registered corp.")
        self._filings = self.company.get_filings(form=['10-K', '10-Q'])
        self._filings_meta_data_df = self.filings.data.to_pandas()
        self._num_xbrl_filings = self.filings_meta_data_df['isXBRL'].count()
        self._cached_financials = None
        self._cached_filings = None
        self._cached_periods = None  # Not a list but a dict
        self._cached_date_duration_maps = None
        self._cached_annual_reporting_dates = None

    def load_financials(self, num_periods: Optional[int] = None, filter_params: dict = None) -> 'CompanyAnalysis':
        """
        Loads financials, using cached data when possible.

        Example:
            # With method chaining
            df = (analysis.load_financials(num_periods=4)
                      .get_income_statement('annual', 4))

        :param num_periods: Number of periods to load (None for all available)
        :param filter_params: Additional filtering parameters for filings
        :return: self: Enables method chaining
        """
        filings = self._get_relevant_filings(num_periods=num_periods, filter_params=filter_params)

        # Check if we can reuse cache
        if not self._can_use_cache(filings):
            # Full reload needed
            self._cached_financials = MultiFinancials(filings)
            self._cached_filings = filings
            self._cached_periods, self._cached_date_duration_maps = self._get_period_dict_and_date_duration_maps()
        return self

    def get_financials(self, frequency: str, num_periods: int, standard: bool = False, return_with_multi_index: bool = False) -> dict:
        """
        Returns a dict(keys='income_statement', 'balance_sheet', 'cash_flow_statement', values=DataFrames) with
        DataFrames containing financial statements.

        Example
            result = get_financials('annual', 4) returns a dict with annual income, cash flow statement and balance
            sheet DataFrames for the past 4 years

        :param frequency: str
        :param num_periods: int
        :param standard: bool
        :param return_with_multi_index: bool (Returns a MultiIndex DataFrame if True, else concept, label and style are
                columns
        :return: dict
        """
        result = {}  # Initialize result

        # Loop through each financial statement and call get_financial
        for statement_name in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
            result[statement_name] = self.get_financial(
                statement_name=statement_name,
                frequency=frequency,
                num_periods=num_periods,
                standard=standard,
                return_with_multi_index=return_with_multi_index
            )
        return result

    def get_financial(self, statement_name: str, frequency: str, num_periods: int, standard: bool = False,
                      return_with_multi_index: bool = False) -> pd.DataFrame:
        """
        Retrieves financial statement data with specified frequency and number of periods.
        Args:
           statement_name: Name of the financial statement (income, balance, cash flow)
           frequency: Reporting frequency (annual, quarterly, ttm)
           num_periods: Number of periods to retrieve
           standard: Only returns a subset of rows (easier to compare across different companies)
                        (Personally I mostly find it useful when I want to extract Net Income or CFO since Net Income
                        varies alot across time both in terms of label and concept and standard = False fails to
                        extract the correct numbers (see GE as an example)).
           return_with_multi_index: If True, returns a DataFrame with label, concept and style as multi index, else
                        these will be columns

        Returns:
           pd.DataFrame: Financial data with accounting item label as index, context as first column and periods as
           columns

        Raises:
           ValueError: If statement_name is not recognized

        Example:
           analysis = CompanyAnalysis('AAPL')
           income = analysis.get_financial('income', 'annual', 5)
        """
        frequency_lower_no_blanks = frequency.lower().replace(' ', '')
        filings_filter_param = {'form': '10-K'} if (frequency_lower_no_blanks in ANNUAL_SYNONYMS) else None

        self.load_financials(num_periods=num_periods, filter_params=filings_filter_param)

        # Load the financial statement into a DataFrame
        statement_name_lower = statement_name.lower().replace(' ', '_')
        if statement_name_lower in BALANCE_SHEET_SYNONYMS:
            financials_df = self._cached_financials.get_balance_sheet(standard=standard).to_dataframe(
                include_format=True,
                include_concept=True)
            act_stmt_name = 'balance_sheet'
        elif statement_name_lower in INCOME_STATEMENT_SYNONYMS:
            financials_df = self._cached_financials.get_income_statement(standard=standard).to_dataframe(
                include_format=True, include_concept=True)
            act_stmt_name = 'income_statement'
        elif statement_name_lower in CASH_FLOW_STATEMENT_SYNONYMS:
            financials_df = self._cached_financials.get_cash_flow_statement(standard=standard).to_dataframe(
                include_format=True, include_concept=True)
            act_stmt_name = 'cash_flow_statement'
        else:
            raise ValueError(f"'{statement_name}' is not a recognized name of a financial statement.")

        # Drop unnecessary columns and store the str columns that are needed later in the index
        financials_df.drop(['decimals', 'level'], axis=1, inplace=True)
        financials_df['label'] = financials_df.index
        financials_df.reset_index(drop=True, inplace=True)
        financials_df.set_index(['concept', 'label', 'style'], inplace=True)

        # Reformat the DataFrame
        financials_df = self._set_sorted_date_column_names(financials_df=financials_df, statement_name=act_stmt_name)

        # Convert data to numeric (all values are str)
        financials_df = make_df_numeric(
            df=financials_df,
        )

        if frequency_lower_no_blanks in ANNUAL_SYNONYMS:
            # In case non-annual cached filings were used before only looking at annual filings, there might be
            # quarterly columns that should be ignored. Only include the report dates that is an annual reporting date
            financials_df = financials_df[
                [c for c in financials_df.columns
                 if (c in list(self._annual_reporting_dates))]  # TODO 'concept'
            ].copy()
        elif frequency_lower_no_blanks in QUARTERLY_SYNONYMS.union(TTM_SYNONYMS):
            # Only calculate quarterly values or trailing twelve months in case of income and cash flow statement
            if act_stmt_name in ['income_statement', 'cash_flow_statement']:
                # Calculate quarterly data if the frequency is not annual
                financials_df = self._calculate_quarterly_data(
                    financial_df=financials_df,
                    date_duration_map=self._cached_date_duration_maps[act_stmt_name]
                )
                if frequency_lower_no_blanks in TTM_SYNONYMS:
                    # Perform a rolling column-wise sum
                    financials_df = columnwise_rolling_sum(df=financials_df, window=4)
        else:
            raise ValueError(f"'{frequency}' is not a recognized frequency")

        # Return the first column ('concept' indexed 0) together with the last columns dictated by num_periods

        # return financials_df.iloc[:, [0] + list(range(-num_periods, 0))]  # TODO 'concept'
        if return_with_multi_index:
            return financials_df.iloc[-num_periods:]
        else:
            data_columns = financials_df.columns[-num_periods:].to_list().copy()
            financials_df.reset_index(inplace=True)
            return financials_df[['concept', 'label', 'style'] + data_columns]

    def get_income_statement(self, frequency: str, num_periods: int, standard: bool = False,
                             return_with_multi_index: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame for income statement data
        DataFrame has accounting items as index, first column is 'context' and the rest are data for each report date in
        ascending order.

        For example
            get_income_statement('annual', 4) returns 4 years of annual data

        :param frequency: str e.g. 'annual' (uses synonyms to find the correct answer)
        :param num_periods: int (larger or equal to 1)
        :param standard: bool
        :param return_with_multi_index: bool (Returns a MultiIndex DataFrame if True, else concept, label and style are
                columns
        :return: DataFrame
        """
        return self.get_financial(statement_name='income_statement', frequency=frequency, num_periods=num_periods,
                                  standard=standard, return_with_multi_index=return_with_multi_index)

    def get_balance_sheet(self, frequency: str, num_periods: int, standard: bool = False, return_with_multi_index: bool = False):
        """
        Returns a DataFrame for balance sheet data
        DataFrame has accounting items as index, first column is 'context' and the rest are data for each report date in
        ascending order.
        :param frequency: str e.g. 'annual' (uses synonyms to find the correct answer)
        :param num_periods: int (larger or equal to 1)
        :param standard: bool
        :param return_with_multi_index: bool (Returns a MultiIndex DataFrame if True, else concept, label and style are
                columns
        :return: DataFrame
        """
        return self.get_financial(statement_name='balance_sheet', frequency=frequency, num_periods=num_periods,
                                  standard=standard, return_with_multi_index=return_with_multi_index)

    def get_cash_flow_statement(self, frequency: str, num_periods: int, standard: bool = False, return_with_multi_index: bool = False):
        """
        Returns a DataFrame for cash flow statement data
        DataFrame has accounting items as index, first column is 'context' and the rest are data for each report date in
        ascending order.
        :param frequency: str e.g. 'annual' (uses synonyms to find the correct answer)
        :param num_periods: int (larger or equal to 1)
        :param standard: bool
        :param return_with_multi_index: bool (Returns a MultiIndex DataFrame if True, else concept, label and style are
                columns
        :return: DataFrame
        """
        return self.get_financial(statement_name='cash_flow_statement', frequency=frequency, num_periods=num_periods,
                                  standard=standard, return_with_multi_index=return_with_multi_index)

    def clear_cache(self):
        """Clears all cached data"""
        self._cached_financials = None
        self._cached_filings = None
        self._cached_periods = None
        self._cached_date_duration_maps = None
        self._cached_annual_reporting_dates = None

    def _calculate_quarterly_data(self, financial_df: pd.DataFrame, date_duration_map: dict) -> pd.DataFrame:
        """
        Returns a DataFrame only including data that has been converted to quarterly values.
        :param financial_df: DataFrame(index=accounting item labels, columns=first is 'context' rest is data with
        report dates (date) as column headers)
        :param date_duration_map: dict(key=date, value=str)
        :return: DataFrame
        """

        financial_df = financial_df.copy()  # Make a copy
        result = {}  # Initialize the result
        report_dates = [c for c in financial_df.columns if not isinstance(c, str)]

        # For each column date, check the duration and which columns if any should be subtracted
        # If date represents 3 month duration, just extract the values. Else, in the case of ...
        # ... 6 months duration
        #       Find dates representing the previous 3 months and subtract
        # ... 9 months duration
        #       Find dates representing the previous 6 months and subtract
        # ... annual duration
        #       Find dates representing the previous 9 months
        #       OR 3 months
        #       and subtract
        for report_date in report_dates:

            duration = date_duration_map[report_date]

            if duration == '3 months':
                # Store the data as it is
                result[report_date] = financial_df.loc[:, report_date]
            else:
                if duration == '6 months':
                    # Subtract ONE data column with 3 month duration or ignore
                    prev_report_dates = self._find_relevant_prev_report_dates(
                        report_date=report_date,
                        date_duration_map=date_duration_map,
                        req_duration='3 months',
                        num_dates=1
                    )
                elif duration == '9 months':
                    # Subtract ONE data column with 6 month duration or ignore
                    prev_report_dates = self._find_relevant_prev_report_dates(
                        report_date=report_date,
                        date_duration_map=date_duration_map,
                        req_duration='6 months',
                        num_dates=1
                    )
                elif duration in ['annual', 'instant']:

                    # Subtract ONE data column with 9 month duration...
                    prev_report_dates = self._find_relevant_prev_report_dates(
                        report_date=report_date,
                        date_duration_map=date_duration_map,
                        req_duration='9 months',
                        num_dates=1
                    )

                    if not prev_report_dates:
                        # ... or THREE data columns with 3 month duration
                        prev_report_dates = self._find_relevant_prev_report_dates(
                            report_date=report_date,
                            date_duration_map=date_duration_map,
                            req_duration='3 months',
                            num_dates=3
                        )
                else:
                    raise ValueError(f'Unexpected duration: {duration}')
                if prev_report_dates is None:
                    continue
                else:
                    # Subtract the relevant previous data to get the quarterly values
                    # This looks messy but I wanted to have the ability to sum data containing some but not all N/A
                    result[report_date] = pd.concat(
                        [
                            financial_df.loc[:, report_date],
                            -financial_df.loc[:, prev_report_dates].sum(axis=1, min_count=1)
                        ], axis=1).sum(axis=1, min_count=1)

        # Store result in a DataFrame and insert the concept column first place
        # result_df = pd.DataFrame(result)
        # result_df.insert(0, 'concept', financial_df['concept'])  # TODO concept (multindex should work)
        return pd.DataFrame(result)

    @staticmethod
    def _find_relevant_prev_report_dates(report_date: date, date_duration_map: dict, req_duration: str,
                                         num_dates: int) -> list | None:
        """
        Find report dates that can be used to subtract data from data at a given report date. I.e. at report date D
        with annual values, one needs to find either 9 month duration data (in the case of most cash flow statements) or
        three 3 month data.
        Returns a list of report dates (datetime.date) taken from a given date_duration_map. Filter out the report date
        that represents a given duration (for example some report dates for Cash Flow have 6 or 9 month duration not 3
        month or annual) that are within one year before the given report date
        :param report_date: date
        :param date_duration_map: dict(keys=date, values=duration str)
        :param req_duration: str e.g. '9 months'
        :param num_dates: int (larger or equal to 1)
        :return:
        """
        if int(num_dates) <= 0:
            raise ValueError(f"num_dates={num_dates} needs to be an int larger or equal to 1")
        # Subtract ONE data column with 9 month duration...
        prev_report_dates = [k for k, v in date_duration_map.items() if
                             v == req_duration.lower() and report_date - timedelta(days=365) < k < report_date]
        prev_report_dates.sort(reverse=True)  # Sorted in descending order and pick the num_dates
        if len(prev_report_dates) < num_dates:
            return None
        else:
            return prev_report_dates[:num_dates]

    def _get_period_dict_and_date_duration_maps(self) -> tuple:
        """
        Returns a tuple with a dict of list of periods (str) in the order based on financials and a dict mapping
        datetime.date to duration of the specific data column in each financial statement
        :return: tuple
            list of period str e.g. '2034' or '31 Jan, 2005'
            dict(keys=datetime.date, value=duration str)
        """
        # Define mapping from internal attribute names to desired output names
        attribute_name_map = {
            'cashflow': 'cash_flow_statement',  # financial has 'cashflow' as attribute not 'cash_flow_statement'
            'income': 'income_statement',  # financial has 'income' as attribute not 'income_statement'
            'balance_sheet': 'balance_sheet'  # stays the same
        }

        # Initialize with the final desired keys
        result = {
            new_name: {}
            for new_name in attribute_name_map.values()
        }
        periods = {
            new_name: []
            for new_name in attribute_name_map.values()
        }
        # For each financial, check the display duration and store it next to the period date str
        for financial in self._cached_financials.financials_list:
            for internal_name, display_name in attribute_name_map.items():
                statement = getattr(financial, internal_name)
                statement_periods = statement.periods
                periods[display_name].extend(statement_periods)
                result[display_name].update(
                    {
                        period: statement.display_duration
                        for period in statement_periods
                    }
                )

        # Rename each date key to be datetime.date instead of str (also handle when str is 'yyyy')
        new_result = {}
        for statement_name in result.keys():
            report_dates = self._get_reportdate_datetime_list(date_str_list=list(result[statement_name].keys()))
            new_result[statement_name] = dict(zip(report_dates, result[statement_name].values()))
        return periods, new_result

    def _can_use_cache(self, requested_filings: List) -> bool:
        """
        Determines if cached data can satisfy the request
        """
        if not self._cached_financials:
            return False

        # Get accession numbers for comparison
        cached_acc_nos = {f.accession_no for f in self._cached_filings}
        requested_acc_nos = {f.accession_no for f in requested_filings}

        # Only reuse if ALL requested filings are in cache
        return requested_acc_nos.issubset(cached_acc_nos)

    def _get_relevant_filings(self, num_periods: Optional[int], filter_params: dict) -> List:
        """Gets filings subset while maintaining order"""

        if num_periods > self._num_xbrl_filings:
            raise ValueError(f"{self._company.display_name}({self.ticker}) has only {self._num_xbrl_filings} filings in XBRL format while the requested number of periods is {num_periods}")
        filings = self.filings
        if filter_params is not None:
            filings = filings.filter(**filter_params)
        if num_periods is None:
            return filings
        return filings.latest(max(num_periods, 4))

    def _set_sorted_date_column_names(self, financials_df: pd.DataFrame, statement_name: str) -> pd.DataFrame:
        """
        Set new column converting str dates to datetime.date. Sorts the columns ascending by date (str column gets
        shifted to the left next to index), converts all dates to datetime.date
        :param financials_df: DataFrame
        :param statement_name: str
        :return: DataFrame
        """

        financials_df = financials_df.copy()

        # For all date str reformat into datetime.date and store into a new list
        new_dates = self._get_reportdate_datetime_list(date_str_list=financials_df.columns)

        # Create a mapping between old date str and new datetime.date
        new_date_map = dict(zip(financials_df.columns, new_dates))

        # Re-order in the way that filings were parsed (this is done to then remove potential duplicates) by first
        # looking at the period str that were seen in each financial filings (cached during download)
        uniq_periods = []  # Only include the unique periods
        for p in self._cached_periods[statement_name]:  # Loop through all periods seen in each financial filings
            if p not in uniq_periods:
                uniq_periods.append(p)
        financials_df = financials_df[uniq_periods].copy()  # Re-order based on the periods

        # Rename columns as datetime.date
        financials_df = financials_df.rename(columns=new_date_map).copy()

        # Remove duplicate columns in case there exists (remove all duplicates except the first leftmost occurrence)
        financials_df = financials_df.loc[:, ~financials_df.columns.duplicated()].copy()

        # Sort ascending with the str column headers at the first columns
        financials_df = financials_df[sorted(financials_df.columns.copy())].copy()
        return financials_df

    def _get_reportdate_datetime_list(self, date_str_list: list) -> list:
        """
        Returns a list of the report dates as datetime.date
        :param date_str_list: list
        :return: list
        """
        # For all date str reformat into datetime.date and store into a new list
        new_dates = []  # Initialize the resulting list
        for d in date_str_list:
            if len(d) != 4:  # d is most likely not a year but a date str like "1 Jan, 2020"
                new_d = dateutil.parser.parse(d).date()
            else:
                # For year str, find the corresponding report date from all the 10-K filings
                filtered_date = self._annual_reporting_dates[
                    pd.to_datetime(self._annual_reporting_dates).dt.year == int(d)]  # Filter based on year
                if len(filtered_date) != 1:
                    raise ValueError(f"No date or several for '{d}' column date")
                new_d = pd.to_datetime(str(filtered_date.values[0])).date()
            new_dates.append(new_d)
        return new_dates

    def handle_duplicate_columns(self, financials_df: pd.DataFrame):
        pass

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
            self._num_xbrl_filings = self.filings_meta_data_df['isXBRL'].count()
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

    @property
    def _annual_reporting_dates(self):
        if self._cached_annual_reporting_dates is None:
            self._cached_annual_reporting_dates = pd.to_datetime(
                self._filings_meta_data_df[self._filings_meta_data_df['form'] == '10-K']['reportDate']).dt.date
        return self._cached_annual_reporting_dates

    def __repr__(self):
        """Gives the precise representation so that the output can be recreated in code"""
        return f"CompanyAnalysis('{self.ticker}')"


def main():
    analyzer = CompanyAnalysis('wmt')

    start = time.time()
    fin_res = analyzer.get_financials(frequency='ttm', num_periods=8)
    print(fin_res)
    print(f"{time.time() - start}")
    start = time.time()
    is_df = analyzer.get_income_statement(frequency='ttm', standard=True, num_periods=8)
    print(is_df)
    print(f"{time.time() - start}")
    start = time.time()
    cf_df = analyzer.get_cash_flow_statement(frequency='ttm', standard=True, num_periods=1, return_with_multi_index=False)
    print(cf_df)
    print(f"{time.time() - start}")

    cf_df.to_clipboard()

    # print('Initialize company')
    # company = Company('WMT')
    # print('Get filings')
    # filings = company.get_filings(form=['10-K', '10-Q']).latest(16)
    # print('Get multi financials')
    # start = time.time()
    # financials = MultiFinancials(filings=filings)
    # print(f"{time.time() - start}")
    # print('Get balance sheet')
    #
    # start = time.time()
    # bs_df = financials.get_balance_sheet(standard=False).to_dataframe(include_concept=True, include_format=True)
    # print(bs_df)
    # print(f"{time.time() - start}")
    #
    # start = time.time()
    # bs_df = financials.get_balance_sheet(standard=True).to_dataframe(include_concept=True, include_format=True)
    # print(bs_df)
    # print(f"{time.time() - start}")
    #
    # start = time.time()
    # bs_df = financials.get_balance_sheet(standard=True).to_dataframe(include_concept=True, include_format=True)
    # print(bs_df)
    # print(f"{time.time() - start}")
    #
    # start = time.time()
    # bs_df = financials.get_balance_sheet(standard=False).to_dataframe(include_concept=True, include_format=True)
    # print(bs_df)
    # print(f"{time.time() - start}")
    #
    # start = time.time()
    # bs_df = financials.get_balance_sheet(standard=False).to_dataframe(include_concept=True, include_format=True)
    # print(bs_df)
    # print(f"{time.time() - start}")

    # print('Get meta data')
    # meta_data = filings.data.to_pandas()


if __name__ == '__main__':
    main()
