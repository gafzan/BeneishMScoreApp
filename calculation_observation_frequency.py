"""calculation_observation_frequency.py"""

import pandas as pd
import re
from typing import Optional, Tuple, List
from datetime import datetime


def add_duration_labels(df: pd.DataFrame):
    """Add duration labels (3mo, 6mo, etc.) to column names"""
    new_cols = {}
    for col in df.columns:
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})', col)
        if len(dates) == 2:
            start_date = datetime.strptime(dates[0], '%Y-%m-%d')
            end_date = datetime.strptime(dates[1], '%Y-%m-%d')
            duration = (end_date - start_date).days / 30  # Approximate months

            # Round to nearest standard duration (3,6,9,12 months)
            standard_duration = min([3, 6, 9, 12], key=lambda x: abs(x - duration))
            new_col = f"{col}__{standard_duration}mo"
            new_cols[col] = new_col

    return df.rename(columns=new_cols)


class ObservationFrequencyDataCalculator:
    """A class for extracting and calculating quarterly and annual financial data from accounting reports.

    This class handles various duration formats and calculates quarterly values using
    a priority-based approach when direct quarterly data is not available.
    """

    def __init__(self, df: pd.DataFrame, report_dates: List[str]):
        """Initialize the Calculations class with accounting data.

        :param df: DataFrame containing accounting data with duration-based column names
        :param report_dates: List containing date str 'YYYY-MM-DD'
        :type df: pd.DataFrame
        """
        self.df = df
        self.report_dates = report_dates

    def _cleaned_data(self) -> pd.DataFrame:
        """Remove columns with all NA values from the DataFrame and drops duplicate columns except the first one
        :return: DataFrame with non-NA columns and no duplicate columns
        :rtype: pd.DataFrame
        """
        result = self.df.dropna(axis=1, how='all').copy()
        # Handle any duplicate col by keeping the first column
        result = result.loc[:, ~result.columns.duplicated(keep='first')].copy()
        return result

    def _check_data(self, df: pd.DataFrame) -> None:
        """
        Raise an error if there is an end date in any duration columns that does not exist as a report date
        :param df: DataFrame
        :return: None
        """
        end_dates = set([col.split('_')[-3] for col in df.columns if 'duration' in col])
        instant_end_dates = set([col.split('_')[-1] for col in df.columns if 'instant' in col])
        end_dates.update(instant_end_dates)
        if any(end_date not in self.report_dates for end_date in end_dates):
            # raise ValueError("There exists End Dates in the duration column headers that does not exist in the Report "
            #                  "Date list")
            pass
            # TODO Could raise a warning
            # E.g. RDDT has first report date at 2024-03-31 however, the data goes further back than that (2023-12-31)
            # THIS IS OK
            # Another example is X that has a weird date (2022-01-01) for some Balance Sheet items
            # However, later each row is selected based on reporting date so not sure it matters ...
        # Todo maybe remove those date columns? Issues came from X and RDDT

    @staticmethod
    def _find_column(df: pd.DataFrame, end_date: str, duration: str) -> Optional[str]:
        """Find column matching the specified end date and duration.

        :param df: DataFrame to search
        :type df: pd.DataFrame
        :param end_date: End date to match in format 'YYYY-MM-DD'
        :type end_date: str
        :param duration: Duration to match (e.g., '3', '6', '9', '12')
        :type duration: str
        :return: Matching column name or None if not found
        :rtype: Optional[str]
        """
        suffix = f"{end_date}__{duration}mo"
        return next((col for col in df.columns if col.endswith(suffix)), None)

    def _get_prev_date(self, idx: int) -> Optional[str]:
        """Get the previous report date for a given index.
        :param idx: Current date index
        :type idx: int
        :return: Previous report date or None if at end of list
        :rtype: Optional[str]
        """
        return self.report_dates[idx + 1] if idx + 1 < len(self.report_dates) else None

    def _try_direct_n_mo(self, df: pd.DataFrame, date: str, n: int) -> Optional[pd.Series]:
        """Attempt to find direct n-month duration data.
        :param df: DataFrame to search
        :type df: pd.DataFrame
        :param date: Report date to match
        :type date: str
        :param n: Number of months
        :type n: int
        :return: Series with 3-month data or None if not found
        :rtype: Optional[pd.Series]
        """
        if col := self._find_column(df, date, f'{n}'):
            return df[col]
        return None

    def _try_annual_minus_3q(self, df: pd.DataFrame, idx: int) -> Optional[pd.Series]:
        """Calculate quarterly value as annual minus previous 3 quarters.

        :param df: DataFrame containing the data
        :type df: pd.DataFrame
        :param idx: Current date index
        :type idx: int
        :return: Calculated quarterly values or None if insufficient data
        :rtype: Optional[pd.Series]
        """
        date = self.report_dates[idx]
        if not (annual_col := self._find_column(df, date, '12')):
            return None

        prev_dates = [d for d in self.report_dates[idx + 1:idx + 4] if d < date]
        prev_cols = [self._find_column(df, d, '3') for d in prev_dates]
        if len(prev_cols) == 3 and all(prev_cols):
            return df[annual_col] - df[prev_cols].sum(axis=1)
        return None

    def _try_subtraction_cases(
            self,
            df: pd.DataFrame,
            idx: int,
            cases: Tuple[Tuple[str, str, str], ...]
    ) -> Optional[pd.Series]:
        """Handle all subtraction-based calculation cases.

        :param df: DataFrame containing the data
        :type df: pd.DataFrame
        :param idx: Current date index
        :type idx: int
        :param cases: Tuple of (current_duration, prev_duration, description) tuples
        :type cases: Tuple[Tuple[str, str, str], ...]
        :return: Calculated values or None if no case matches
        :rtype: Optional[pd.Series]
        """
        date = self.report_dates[idx]
        prev_date = self._get_prev_date(idx)
        if not prev_date:
            return None

        for current_dur, prev_dur, _ in cases:
            if (current_col := self._find_column(df, date, current_dur)) and \
                    (prev_col := self._find_column(df, prev_date, prev_dur)):
                return df[current_col] - df[prev_col]
        return None

    def get_annual_data(self, periods: int) -> pd.DataFrame:
        """Extract annual data.
        :param periods: Number of periods to calculate
        :type periods: int
        :return: DataFrame with annual values
        :rtype: pd.DataFrame
        """
        clean_df = self._cleaned_data()
        self._check_data(df=clean_df)

        duration_cols = [col for col in clean_df.columns if 'duration' in col]
        instant_cols = [col for col in clean_df.columns if 'instant' in col]

        # Split the DataFrame
        duration_df = clean_df[duration_cols].dropna(axis=0, how='all').copy()
        instant_df = clean_df[instant_cols].dropna(axis=0, how='all').copy()

        duration_result_df = pd.DataFrame(index=duration_df.index)

        for i in range(min(periods, len(self.report_dates))):
            current_date = self.report_dates[i]

            # Try each method in order until we get a non-None value
            if (val := self._try_direct_n_mo(clean_df, current_date, 12)) is not None:
                value = val
            else:
                value = None

            duration_result_df[current_date] = value if value is not None else pd.NA

        # Get the instant data
        instant_result_df = self._get_relevant_instant_data(instant_df=instant_df, periods=periods)

        # Combine the duration accounting items with the instant ones
        result_df = pd.concat([duration_result_df, instant_result_df], axis=0).reindex(clean_df.index)
        return result_df

    def _get_relevant_instant_data(self, instant_df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """
        Return a DataFrame(index=accounting items, cols=report dates)
        :param instant_df: DataFrame with instant accounting items e.g. Total Assets
        :param periods: int
        :return: DataFrame
        """
        if instant_df.empty:
            return instant_df.copy()
        relevant_dates = self.report_dates[:periods].copy()
        col_date_map = {
            f'instant_{d}': d
            for d in relevant_dates
        }
        result_df = instant_df[list(col_date_map.keys())].copy()
        result_df.rename(columns=col_date_map, inplace=True)
        return result_df

    def get_quarterly_data(self, periods: int) -> pd.DataFrame:
        """Extract quarterly data using priority-based approach.

        Priority order:
        1. Direct 3-month data
        2. Annual minus previous 3 quarters
        3. Annual minus 9-month (previous period)
        4. 9-month minus 6-month (previous period)
        5. 6-month minus 3-month (previous period)

        :param periods: Number of periods to calculate
        :type periods: int
        :return: DataFrame with quarterly values
        :rtype: pd.DataFrame
        """
        clean_df = self._cleaned_data()
        self._check_data(df=clean_df)

        duration_cols = [col for col in clean_df.columns if 'duration' in col]
        instant_cols = [col for col in clean_df.columns if 'instant' in col]

        # Split the DataFrame
        duration_df = clean_df[duration_cols].dropna(axis=0, how='all').copy()
        instant_df = clean_df[instant_cols].dropna(axis=0, how='all').copy()

        duration_result_df = pd.DataFrame(index=duration_df.index)

        subtraction_cases = (
            ('12', '9', 'annual_minus_9mo'),  # Priority 3
            ('9', '6', '9mo_minus_6mo'),  # Priority 4
            ('6', '3', '6mo_minus_3mo')  # Priority 5
        )

        for i in range(min(periods, len(self.report_dates))):
            current_date = self.report_dates[i]

            # Try each method in order until we get a non-None value
            if (val := self._try_direct_n_mo(duration_df, current_date, 3)) is not None:
                value = val
            elif (val := self._try_annual_minus_3q(duration_df, i)) is not None:
                value = val
            else:
                value = self._try_subtraction_cases(duration_df, i, subtraction_cases)

            duration_result_df[current_date] = value if value is not None else pd.NA

        # Get the instant data
        instant_result_df = self._get_relevant_instant_data(instant_df=instant_df, periods=periods)

        # Combine the duration accounting items with the instant ones
        result_df = pd.concat([duration_result_df, instant_result_df], axis=0).reindex(clean_df.index)
        return result_df

    def __repr__(self):
        return f"QuarterlyDataCalculator(DataFrame(rows={self.df.shape[0]}, cols={self.df.shape[1]}), {len(self.report_dates)} report dates)"


