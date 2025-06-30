"""accounting_item_extractor.py"""

import pandas as pd
import re
from typing import Dict, Literal, Union, List as TypingList, Tuple as TypingTuple, Optional
from functools import reduce
import operator

from accounting_item_filter_config import AccountingItemFilterConfig

import logging

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class AccountingItemExtractor:
    def __init__(self, df: pd.DataFrame = None):
        self._df = df.copy() if df is not None else None
        self.prefixes = ['us-gaap_', 'us-gaap:']

    def add_prefixes_suffixes_to_ignore(self, ignore: str | TypingList[str]) -> None:
        """Adds whatever it is to be ignored when cleaning str e.g. could involve a ticker"""
        ignore = [ignore] if isinstance(ignore, str) else ignore
        self.prefixes.extend(ignore)

    def _clean_text(self, text: str) -> str:
        """Remove common prefixes and normalize concept or label name"""
        for prefix in self.prefixes:
            text = text.replace(prefix, '')
        return text.lower().strip()

    def _exact_match(self, match_by: Literal['label', 'concept'], search_term: str) -> pd.DataFrame | None:
        """Exact concept match (prefix-insensitive)"""
        if match_by not in ['label', 'concept']:
            raise ValueError("'match_by' can only be set to either 'label' or 'concept'")
        matches = self.df[
            self.df[match_by].apply(self._clean_text) == search_term
            ].copy()
        if not matches.empty:
            logger.info(f"Found {matches.shape[0]} row(s) exact {match_by} match for search term '{search_term}'")
            return matches
        else:
            return None

    def _exact_match_from_items(self, match_by: Literal['label', 'concept'], filter_config: Dict[str, TypingList[str]]) -> pd.DataFrame | None:
        """Exact concept or label match from common items"""
        if match_by not in ['label', 'concept']:
            raise ValueError("'match_by' can only be set to either 'label' or 'concept'")

        if filter_config:
            exact_matches = self.df[
                self.df[match_by].apply(self._clean_text).isin(
                    [x.lower() for x in filter_config.get(f'exact_{match_by}s', [])]
                )
            ].copy()
            if not exact_matches.empty:
                logger.info(f"Found {exact_matches.shape[0]} row(s) exact {match_by} match from filter config.")
                return exact_matches

    def _partial_match(
            self,
            match_by: Union[Literal['label', 'concept'], TypingList[Literal['label', 'concept']]],
            filter_config: Dict[str, TypingList[str]],
            join_columns_with_or: bool = False  # New parameter to control AND/OR between columns
    ) -> pd.DataFrame | None:
        """
        Partial match supporting:
        - AND/OR logic for both inclusions and exclusions
        - Single column or multiple column matching
        - Vectorized operations for performance
        - New parameter join_columns_with_or: if True, uses OR between columns, otherwise AND (default)
        """

        # Input validation and setup
        if isinstance(match_by, str):
            match_by = [match_by]
        if not all(col in ['label', 'concept'] for col in match_by):
            raise ValueError("match_by must be 'label', 'concept' or list containing both")

        # Process includes and excludes for all specified columns
        include_masks, exclude_masks = [], []  # Initialize mask lists

        # Loop through each match by columns (label or concept)
        for column in match_by:
            if partial_key := f'partial_{column}s':
                include_terms = filter_config.get(partial_key, [])
                if include_terms:
                    include_masks.append(self._process_column(column, include_terms))

            if exclude_terms := filter_config.get('partial_exclusions', []):
                exclude_masks.append(self._process_column(column, exclude_terms))

        # Combine results across columns (AND or OR between columns based on parameter)
        if join_columns_with_or:
            include_mask = reduce(operator.or_, include_masks) if include_masks else pd.Series(False,
                                                                                               index=self.df.index)
        else:
            include_mask = reduce(operator.and_, include_masks) if include_masks else pd.Series(False,
                                                                                                index=self.df.index)

        exclude_mask = reduce(operator.or_, exclude_masks) if exclude_masks else pd.Series(False, index=self.df.index)

        matches = self.df[include_mask & ~exclude_mask].copy()

        if not matches.empty:
            logger.info(
                f"Found {matches.shape[0]} partial matches across '%s' from filter config." % "' and '".join(match_by))
        return matches if not matches.empty else None

    def _process_column(self, column: str, terms: list) -> pd.Series:
        """Helper function to process terms for a single column"""
        cleaned = self.df[column].apply(self._clean_text)
        masks = []
        for term in terms:
            if isinstance(term, (list, tuple)):  # AND logic
                masks.append(reduce(operator.and_, (cleaned.str.contains(t.lower()) for t in term)))
            else:  # OR logic
                masks.append(cleaned.str.contains(term.lower()))
        return reduce(operator.or_, masks, pd.Series(False, index=self.df.index))

    def _regex_match(self, match_by: Literal['label', 'concept'], search_term: str) -> pd.DataFrame | None:
        """Regular expression match based on either label or concept"""
        if match_by not in ['label', 'concept']:
            raise ValueError("'match_by' can only be set to either 'label' or 'concept'")

        regex_matches = self.df[
            self.df[match_by].apply(self._clean_text).str.contains(
                r'\b' + re.escape(search_term) + r'\b', regex=True
            )
        ].copy()
        if not regex_matches.empty:
            logger.info(f"Found {regex_matches.shape[0]} row(s) regex {match_by} match for search term '{search_term}'")
            return regex_matches

    def _sequence_matches(self, filter_config: Dict[str, TypingList[str]]):

        sequence_config = {
            'single_concept_sequence': {
                'match_by': 'concept',
                'single': True
            },
            'single_label_sequence': {
                'match_by': 'label',
                'single': True
            },
            'non_empty_concept_sequence': {
                'match_by': 'concept',
                'single': False
            },
            'non_empty_label_sequence': {
                'match_by': 'label',
                'single': False
            },
        }

        # Check so that these are valid AccountingItemFilterConfig (might have changed names)
        matches = None
        for key in sequence_config.keys():
            if key not in AccountingItemFilterConfig.model_json_schema()['properties'].keys():
                raise ValueError(f"'{key}' is not a valid attribute of {AccountingItemFilterConfig.__name__}")

            if key in filter_config:
                matches = self._sequence_match(
                    match_by=sequence_config[key]['match_by'],
                    single=sequence_config[key]['single'],
                    sequence=filter_config[key],
                    excluding_terms=filter_config.get('partial_exclusions', None)
                )
                if matches is not None:
                    return matches
        return matches

    def _sequence_match(self,
                      match_by: Literal['label', 'concept'],
                      single: bool,
                      sequence: TypingList[Union[str, TypingList[str], TypingTuple[str]]],
                      excluding_terms: Optional[TypingList[Union[str, TypingList[str], TypingTuple[str]]]]
                      ) -> pd.DataFrame:
        """
        Optimized sequence matching with exclusion terms support.

        Args:
            match_by: Column to search ('label' or 'concept')
            single: If True, requires exactly one match
            sequence: Ordered search terms (str or compound terms)
            excluding_terms: Terms that disqualify matches (str or compound terms)

        Returns:
            Matching DataFrame or None if no match found
        """
        df = self.df.copy()
        df[match_by] = df[match_by].apply(self._clean_text)
        col = df[match_by].str

        # Create exclusion mask if terms are provided
        exclude_mask = pd.Series(False, index=df.index)
        if excluding_terms:
            for term in excluding_terms:
                if isinstance(term, (list, tuple)):
                    term_masks = [col.contains(t, case=False, na=False) for t in term]
                    exclude_mask |= reduce(operator.and_, term_masks)
                else:
                    exclude_mask |= col.contains(term, case=False, na=False)

        for element in sequence:
            if isinstance(element, (list, tuple)):
                # Create masks for all terms and combine with AND
                masks = [col.contains(term, case=False, na=False) for term in element]
                mask = reduce(operator.and_, masks)
            else:
                mask = col.contains(element, case=False, na=False)

            # Apply exclusion filter
            if excluding_terms:
                mask &= ~exclude_mask

            filtered = df[mask]

            if not filtered.empty:
                if not single or len(filtered) == 1:
                    return filtered

        return None

    def extract_accounting_items(self, search_term: str = None, filter_config: Dict[str, TypingList[str]] = None,
                                 filter_by_total: bool = True, use_regex: bool = False) -> pd.DataFrame:
        """
        Extract DataFrame rows by filtering either by a str input ('search_term') or by a filter configuration dict
        ('filter_config').

        Example:
            filter_config = {
                'exact_concepts': ['CostOfRevenue', 'CostOfGoodsSold'],
                'partial_concepts': ['costof', 'cogs', 'expense'],
                'partial_labels': ['cost of goods', 'cogs', 'expense'],
                'partial_exclusions': ['interest', 'operating', 'depreciation', 'amortization', 'marketing', 'general',
                                       'admin', 'stock', 'write', 'occurring', 'tax', 'research', 'development',
                                       'r&d', 'lease'],
            }
            attempts to filter Cost of Goods Sold ('COGS')

        :param search_term: str
        :param filter_config: dict(keys='exact_concepts', 'exact_labels', 'partial_concepts', 'partial_labels', 'partial_exclusions', values=list of str)
                                For more details you can call AccountingItemFilterConfig.print_input_structure()
        :param filter_by_total: bool If True, a final filter is performed to only include rows where style = 'Total'
                                If there are no rows with 'Total' return the rows as if nothing was filtered by Total
        :param use_regex: bool (only when search_term is specified else warning) in case no match is found in concept or
                                label check using regular expression on both concept and label
        :return: DataFrame
        """
        if self.df is None:
            raise ValueError("Specify a DataFrame in the df attribute")

        if (search_term is None) == (filter_config is None):
            raise ValueError("Provide either search_term or filter_config, not both")

        if search_term:
            # First search for exact match among concepts, then labels
            clean_search_term = self._clean_text(text=search_term)
            match_method_configs = [
                (self._exact_match, {'match_by': 'concept', 'search_term': clean_search_term}),
                (self._exact_match, {'match_by': 'label', 'search_term': clean_search_term})
            ]
            if use_regex:
                # Use regex on concept and labels
                match_method_configs.extend(
                    [
                        (self._regex_match, {'match_by': 'concept', 'search_term': clean_search_term}),
                        (self._regex_match, {'match_by': 'label', 'search_term': clean_search_term})
                    ]
                )
        else:
            # Use filter configuration
            # Make sure the filter config has the correct schema
            AccountingItemFilterConfig.validate_input_structure(filter_config=filter_config)

            if use_regex:
                logger.warning("'use_regex' has no effect when specifying 'filter_config'")

            if filter_config.get('concept_label_union', False):
                match_method_configs = [
                    (self._exact_match_from_items, {'match_by': 'concept', 'filter_config': filter_config}),  # Level 1: Exact concept match from common items
                    (self._exact_match_from_items, {'match_by': 'label', 'filter_config': filter_config}),  # Level 2: Exact label match from common items
                    (self._sequence_matches, {'filter_config': filter_config}),  # Level 3: Sequential matches
                    (self._partial_match, {'match_by': ['concept', 'label'], 'filter_config': filter_config, 'join_columns_with_or': True}),  # Level 4: Partial match with concept OR label with excl.
                ]
            else:
                match_method_configs = [
                    (self._exact_match_from_items, {'match_by': 'concept', 'filter_config': filter_config}),  # Level 1: Exact concept match from common items
                    (self._exact_match_from_items, {'match_by': 'label', 'filter_config': filter_config}),  # Level 2: Exact label match from common items
                    (self._sequence_matches, {'filter_config': filter_config}),  # Level 3: Sequential matches
                    (self._partial_match, {'match_by': ['concept', 'label'], 'filter_config': filter_config}),  # Level 4: Partial match with concept AND label with excl.
                    (self._partial_match, {'match_by': 'concept', 'filter_config': filter_config}),  # Level 5: Partial concept match with exclusions
                    (self._partial_match, {'match_by': 'label', 'filter_config': filter_config}),  # Level 6: Partial label match from common items
                ]

        matches = None  # Initialize the matches
        for match_method_config in match_method_configs:  # Loop through the matching configs
            matches = match_method_config[0](**match_method_config[1])
            if matches is not None:
                break

        # Returns an empty DataFrame if no matches were found
        if matches is None:
            return pd.DataFrame(columns=self.df.columns)

        # If there are multiple matches, pick the one that has Total if there is one
        if filter_by_total and matches.shape[0] > 1 and matches['is_total'].any():
            logger.info(f"Filter based on 'Total' (rows before = {matches.shape[0]})")
            matches = matches[matches['is_total']].copy()
            logger.info(f"Filter based on 'Total' (rows after = {matches.shape[0]})")

        return matches

    def __repr__(self):
        return f"AccountingDataExtractor({self.df}, {self.prefixes})"

    def __str__(self):
        return f"AccountingDataExtractor(DataFrame({self.df.shape[0]} rows × {self.df.shape[1]} cols), {self.prefixes})"

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame):
        if df is not None:
            self._df = df.copy()
        else:
            self._df = None






