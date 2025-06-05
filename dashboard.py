"""dashboard.py"""

import pandas as pd
import numpy as np
import streamlit as st
from company_analysis import CompanyAnalysis
from accounting_data_extractor import AccountingDataExtractor
from edgar import set_identity


# First define all the constants
ACCOUNTS_RECEIVABLE_NAME = "Account Receivable"
CURRENT_ASSETS_NAME = "Current Assets"
CURRENT_LIABILITIES_NAME = "Current Liabilities"
TOTAL_ASSET_NAME = "Total Assets"
PPE_NET_NAME = "PPE (Net)"
LONG_TERM_DEBT_NAME = "Long Term Debt"
SALES_NET_NAME = "Revenues (Net)"
DEPRECIATION_AMORTIZATION_NAME = "D&A Expense"
COGS_NAME = "Cost of Goods Sold"
SGA_EXPENSE_NAME = "SG&A Expense"
CFO_NAME = "Cash From Operations"
NET_INCOME_NAME = "Net Income (excl. Extr. Items)"

# Then build the config using the constants
ACCOUNTING_ITEM_CONFIG = {
    ACCOUNTS_RECEIVABLE_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_accounts_receivable_items.__name__,
        'financial_statement_name': ['standard_balance_sheet', 'balance_sheet'],
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Amounts due from customers for goods/services delivered but not yet paid, net of allowances",
    },
    CURRENT_ASSETS_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_current_asset_items.__name__,
        'financial_statement_name': 'balance_sheet',
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Assets expected to be converted to cash or used within one year (cash, inventory, receivables)",
    },
    CURRENT_LIABILITIES_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_current_liabilities_items.__name__,
        'financial_statement_name': 'balance_sheet',
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Obligations due within one year (payables, short-term debt, accrued expenses)",
    },
    TOTAL_ASSET_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_total_assets_items.__name__,
        'financial_statement_name': 'balance_sheet',
        'non_negative': True,
        'zero_leads_to_error': True,
        'help': "Sum of all resources owned by the company (current + long-term assets)",
    },
    PPE_NET_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_property_plant_equipment_items.__name__,
        'financial_statement_name': 'balance_sheet',
        'non_negative': True,
        'zero_leads_to_error': True,
        'help': "Property, plant and equipment net of accumulated depreciation (tangible long-term assets)",
    },
    LONG_TERM_DEBT_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_long_term_debt_items.__name__,
        'financial_statement_name': 'balance_sheet',
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Borrowings with maturities beyond one year (bonds, loans, mortgages)",
    },
    SALES_NET_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_revenue_items.__name__,
        'financial_statement_name': 'income_statement',
        'non_negative': True,
        'zero_leads_to_error': True,
        'help': "Total revenues from goods sold/services rendered, net of returns and allowances",
    },
    DEPRECIATION_AMORTIZATION_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_depreciation_amortization_items.__name__,
        'financial_statement_name': ['cash_flow_statement', 'income_statement'],
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Depreciation & amortization expense allocates the costs of an asset over their useful lives",
    },
    COGS_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_cogs_items.__name__,
        'financial_statement_name': 'income_statement',
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Direct costs attributable to production of goods sold (materials, labor, overhead)",
    },
    SGA_EXPENSE_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_sales_general_admin_items.__name__,
        'financial_statement_name': 'income_statement',
        'non_negative': True,
        'zero_leads_to_error': False,
        'help': "Sales, general and administrative expenses (salaries, marketing, office costs)",
    },
    CFO_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_cash_from_operations_items.__name__,
        'financial_statement_name': ['standard_cash_flow_statement', 'cash_flow_statement'],
        'non_negative': False,
        'zero_leads_to_error': False,
        'help': "Cash generated from core business operations (before investing/financing activities)",
    },
    NET_INCOME_NAME: {
        'extract_method_name': AccountingDataExtractor.extract_net_income_items.__name__,
        'financial_statement_name': ['standard_income_statement', 'income_statement'],
        'non_negative': False,
        'zero_leads_to_error': False,
        'help': "Profit after all expenses and taxes, excluding one-time/unusual items",
    },
}

PREV_YEAR_COL_NAME = 'Previous year'
CURRENT_YEAR_COL_NAME = 'Current year'

WINSORIZE_VALUES = {
    'DSRI': (0.27, 3.12),
    'GMI': (0.49, 4.880),
    'AQI': (0.1, 4.09),
    'SGI': (0.49, 4.88),
    'DEPI': (0.36, 2.65),
    'SGAI': (0.37, 2.19),
    'LVGI': (0.17, 3.13),
    'TATA': (-0.6, 0.25)
}

st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
    width: 80vw;
    height: 95vh;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_accounting_item_names() -> list:
    return list(ACCOUNTING_ITEM_CONFIG.keys())


def get_non_negative_accounting_item_names() -> list:
    return [k for k, v in ACCOUNTING_ITEM_CONFIG.items() if v['non_negative']]


def get_non_zero_accounting_item_names() -> list:
    return [k for k, v in ACCOUNTING_ITEM_CONFIG.items() if v['zero_leads_to_error']]


def initialize_session_state():
    """Initialize all required session state keys"""
    if 'analysis' not in st.session_state:
        st.session_state.analysis = {}

    if 'accounting_inputs' not in st.session_state:
        st.session_state.accounting_inputs = {}

    if 'winsorize' not in st.session_state:
        st.session_state.winsorize = True

    if 'in_millions' not in st.session_state:
        st.session_state.in_millions = True  # Default to millions

    # Initialize all input keys
    default_df = get_default_df()
    for index_item in default_df.index:
        for col in default_df.columns:
            key = f"{index_item.lower()}_{col.lower().replace(' ', '_')}"
            input_key = f"input_{key}"
            if input_key not in st.session_state:
                st.session_state[input_key] = ""


@st.dialog("Set your SEC identifier")
def set_your_identity():
    """Opens up a dialog window asking user to input an identifier"""
    st.write("Enter an identity that the SEC can use to track their traffic.\nI usually put my mail but I think a name is fine.")
    identifier = st.text_input("🥸Enter an identifier here...", placeholder="🥸Enter an identifier here...",
                               label_visibility='collapsed')
    if st.button("Submit"):
        st.session_state['identifier'] = identifier
        st.rerun()


def get_default_df() -> pd.DataFrame:
    """Returns a DataFrame with default empty values"""
    return pd.DataFrame(
        {
            PREV_YEAR_COL_NAME: 12 * [''],
            CURRENT_YEAR_COL_NAME: 12 * ['']
        },
        index=get_accounting_item_names()
    )


def on_ticker_change():
    """Handle ticker change event"""
    ticker = st.session_state.ticker_input.upper()

    if ticker:

        if 'identifier' not in st.session_state:
            set_your_identity()
        else:
            set_identity(user_identity=st.session_state.identifier)

            try:
                if ticker not in st.session_state.analysis:

                    st.session_state.analysis[ticker] = CompanyAnalysis(ticker=ticker)
                df = get_accounting_input_data(ticker=ticker)
                set_session_state_accounting_items_from_df(df=df)
                unit = " (USD millions)" if st.session_state.get('in_millions', True) else ""
                st.success(f"Financial data loaded for {ticker}: Trailing twelve months{unit}")
            except ValueError as e:
                st.error(e, icon="🚨")
                clear_data()
                return
    else:
        clear_data()


def clear_data():
    """Clear all input data"""
    st.session_state.ticker_input = ""
    default_df = get_default_df()
    set_session_state_accounting_items_from_df(df=default_df)


def set_session_state_accounting_items_from_df(df: pd.DataFrame):
    """Update session state with values from DataFrame"""
    for index_item in df.index:
        for col in df.columns:
            key = f"{index_item.lower()}_{col.lower().replace(' ', '_')}"
            input_key = f"input_{key}"
            st.session_state[input_key] = str(df.loc[index_item, col])


def get_accounting_input_data(ticker: str) -> pd.DataFrame:
    """Get accounting data for ticker"""
    if ticker:
        if ticker not in st.session_state.accounting_inputs:
            # Use CompanyAnalysis to fetch data here (should only be done ONCE for each ticker)
            st.session_state.accounting_inputs[ticker] = get_accounting_data_from_sec(ticker=ticker)

        df = st.session_state.accounting_inputs[ticker].copy()

        # Convert to millions if the toggle is on
        if st.session_state.get('in_millions', True):
            num_cols = [col for col in df.columns if col not in ['concept', 'label', 'style']]
            df.loc[:, num_cols] = df.loc[:, num_cols] / 1e6
            df = df.round(2)  # Round to 2 decimals

        return df
    else:
        return get_default_df()


@st.cache_data
def get_raw_sec_data(ticker: str) -> dict:
    """

    :param ticker: str
    :return: dict of 5 DataFrames
    """

    analyst = st.session_state.analysis[ticker]

    # Load income statement, cash flow statement and balance sheet for the past 8 periods with trailing twelve months
    data = analyst.get_financials(frequency='ltm', num_periods=8)
    standard_data = analyst.get_financials(frequency='ltm', num_periods=8, standard=True)
    data.update(
        {f'standard_{k}': v for k, v in standard_data.copy().items()}
    )
    data = clean_raw_sec_data(sec_results=data)
    return data


def clean_raw_sec_data(sec_results: dict) -> dict:
    """
    Returns the same dict(keys=financial statement name, values=DataFrames) after dividing each value by 1e6 and remove
    rows that probably is shares or values per share
    :param sec_results: dict
    :return: dict
    """
    result = {}
    extractor = AccountingDataExtractor()
    for fs_name, df in sec_results.items():
        df = df.copy()

        if 'income_statement' in fs_name:  # 'income_statement' and 'standard_income_statement'
            extractor.df = df
            df = extractor.extract_accounting_items(
                filter_config={
                    'partial_concepts': [''],
                    'partial_labels': [''],
                    'partial_exclusions': ['per share', 'earningspershare', 'perbasicshare', 'perdilutedshare',
                                           'NumberOfSharesOutstanding', ' eps ', ('numberof', 'shares')]
                },
                filter_by_total=False
            )

        result[fs_name] = df
    return result


@st.cache_data
def get_accounting_data_from_sec(ticker: str) -> pd.DataFrame:
    """
    Returns a DataFrame with all the accounting item names as index and two columns (previous and current year).
    :param ticker: str
    :return: DataFrame
    """

    # First retrieve the data (cache are already used so no need to store in st)
    data = get_raw_sec_data(ticker=ticker)

    # Initialize the data extractor
    extractor = AccountingDataExtractor()
    extractor.add_prefixes_suffixes_to_ignore(f'{ticker}_')
    result = []
    for item_name, acc_item_config in ACCOUNTING_ITEM_CONFIG.items():

        financial_statement_names = acc_item_config['financial_statement_name'] if isinstance(acc_item_config['financial_statement_name'], list) \
            else [acc_item_config['financial_statement_name']]
        filtered_df = None
        for financial_statement_name in financial_statement_names:

            extractor.df = data[financial_statement_name]  # Give the data to the extractor
            filtered_df = getattr(extractor, acc_item_config['extract_method_name'])()  # Call the correct extractor method
            if filtered_df.empty:
                continue
            else:
                break

        # Drop str columns and do a column-vise sum (needs to have at least one value else nan)
        account_item_df = filtered_df.drop(['concept', 'label', 'style'], axis=1).sum(axis=0, min_count=1).to_frame()
        account_item_df.columns = [item_name]
        result.append(
            account_item_df
        )

    # Concatenate the result into one big DataFrame and extract only the necessary columns and handle negative values
    return concat_and_format_sec_result(sec_df_list=result)


def concat_and_format_sec_result(sec_df_list: list) -> pd.DataFrame:
    """
    Concatenate all the DataFrames specified in a list, transpose, select latest and one year old values and change the
    colum names. Also all negative values when applicable (not for net income or CFO) changes to positive
    :param sec_df_list: list of DataFrames
    :return: DataFrame
    """

    # Concatenate the result into one big DataFrame and extract only the necessary columns
    result_df = pd.concat(sec_df_list, axis=1).T.iloc[:, [-5, -1]]
    result_df.columns = [PREV_YEAR_COL_NAME, CURRENT_YEAR_COL_NAME]

    # Change sign of negative values
    non_neg_acc_items = get_non_negative_accounting_item_names()
    result_df.loc[non_neg_acc_items] = result_df.loc[non_neg_acc_items].abs()
    return result_df


def display_wiki():
    """Display information about M-Score"""
    with st.expander("Wiki 📚"):
        st.subheader("Beneish M-Score")
        st.write(
            "The Beneish M-Score is a numerical formula that aims to identify companies that are engaging in earnings manipulation.")

        st.markdown("**M-Score Formula:**")
        st.latex(r'''
            \begin{aligned}
            M\text{-}Score &= -4.84 \\
            &\quad + 0.920 \times DSRI \\
            &\quad + 0.528 \times GMI \\
            &\quad + 0.404 \times AQI \\
            &\quad + 0.892 \times SGI \\
            &\quad + 0.115 \times DEPI \\
            &\quad - 0.172 \times SGAI \\
            &\quad + 4.679 \times TATA \\
            &\quad - 0.327 \times LVGI
            \end{aligned}
        ''')
        st.subheader("Additional sources")


def user_guide():
    """Display a user guide on the sidebar"""
    with st.expander("User Guide ❔", expanded=False):
        st.markdown("""
        ## How to Use This Tool

        ### 1. Enter Ticker Symbol
        - Input a valid U.S. stock ticker (e.g., AAPL for Apple, Inc.)
        - Data will automatically download from SEC Edgar

        ### 2. Review/Edit Data
        - The app pre-fills financial data from SEC filings (latest trailing twelve month)
        - Manually verify or edit any values as needed
        - Use the "Hand-pick data" button to select specific line items

        ### 3. Calculate M-Score
        - Click "Calculate" to run the analysis
        - Results show:
          - 🟢 Score ≤ -1.78: Low manipulation risk
          - 🟡 Score -1.78 to -2: Possible manipulation
          - 🔴 Score > -1.78: High manipulation risk

        ### Key Features
        - **Winsorize Option**: Limits extreme values per Beneish's methodology
        - **Data Validation**: Checks for negative/zero values where inappropriate
        - **Detailed Breakdown**: Shows all 8 components of the M-Score

        ### About the Beneish M-Score
        The M-Score identifies earnings manipulation using 8 financial ratios (see Wiki for details):
        1. DSRI - Days Sales in Receivables Index
        2. GMI - Gross Margin Index  
        3. AQI - Asset Quality Index
        4. SGI - Sales Growth Index
        5. DEPI - Depreciation Index
        6. SGAI - SG&A Expense Index
        7. LVGI - Leverage Index
        8. TATA - Total Accruals to Assets

        ### Interpretation
        - **Threshold**: -1.78
        - Scores above -1.78 suggest higher probability of manipulation
        - Scores below -2 suggest lower probability

        ### Tips
        - Investigate any score > -1.78 in detail
        - Use with other fundamental analysis tools
        """)

        st.markdown("""
        <style>
        .user-guide {
            font-size: 0.9em;
            line-height: 1.6;
        }
        .user-guide h3 {
            color: #2e86ab;
            margin-top: 1em;
        }
        .user-guide ul {
            padding-left: 1.5em;
        }
        </style>
        """, unsafe_allow_html=True)


def get_all_selected_accounting_item_names() -> list:
    """
    Returns a list of str of all the accounting item names that has been selected in a checkbox
    :return: list of str
    """
    selected_items = []
    for index_item in get_accounting_item_names():
        select_key = f"select_{index_item.lower().replace(' ', '_')}"
        if select_key in st.session_state and st.session_state[select_key]:
            selected_items.append(index_item)
    return selected_items


@st.dialog("Hand-pick Accounting Data")
def select_data_manually():
    """Popup for selecting and aggregating accounting data"""
    selected_items = get_all_selected_accounting_item_names()
    if not selected_items:
        return st.warning("No items selected in main form")

    # Initialize session states
    st.session_state.setdefault('selected_rows', {})
    st.session_state.setdefault('aggregated_values', {})

    tabs = st.tabs(selected_items)
    for tab, item in zip(tabs, selected_items):
        with tab:
            st.subheader(f"Select rows for: {item}")
            st.write("When done with the selection press 'Confirm New Values'")
            fs_name = next(v['financial_statement_name'] for k, v in ACCOUNTING_ITEM_CONFIG.items() if k == item)

            if isinstance(fs_name, list):
                fs_name = st.selectbox(
                    f"Select financial statement for {item}",
                    options=[n.replace('_', ' ').title() for n in fs_name],
                    key=f"fs_select_{item}"
                )
                fs_name = fs_name.replace(' ', '_').lower()

            df = get_raw_sec_data(st.session_state.ticker_input.upper())[fs_name]
            df.columns = [c.strftime('%Y-%m-%d') if c not in ['concept', 'label', 'style']
                          else c for c in df.copy().columns]
            df = df.iloc[:, [0, 1, 2, -5, -1]].copy()

            # Convert to millions if the toggle is on
            if st.session_state.get('in_millions', True):
                df.iloc[:, [-2, -1]] = df.iloc[:, [-2, -1]] / 1e6
                df = df.round(2)  # Round to 2 decimals

            all_cols = df.columns.tolist()
            df.insert(0, 'Select', False)

            edited_df = st.data_editor(
                df,
                disabled=all_cols,
                column_config={"Select": st.column_config.CheckboxColumn("Select")},
                hide_index=True,
                key=f"data_editor_{item}"
            )

            selected = edited_df[edited_df.Select].index.tolist()
            st.session_state.selected_rows[item] = selected

            if selected:
                numeric_cols = [c for c in df.columns if c not in ['concept', 'label', 'style']]
                agg = edited_df.loc[selected, numeric_cols].sum()
                st.session_state.aggregated_values[item] = {
                    PREV_YEAR_COL_NAME: agg.iloc[-2],
                    CURRENT_YEAR_COL_NAME: agg.iloc[-1]
                }

                # Create a horizontal layout for metrics + button
                col1, col2, col3 = st.columns([1, 1, 2])  # Adjust ratios as needed

                with col1:
                    st.metric(
                        label=PREV_YEAR_COL_NAME,
                        value=f"{agg.iloc[-2]:,.0f}",
                        help=f"Aggregated value for {PREV_YEAR_COL_NAME}"
                    )

                with col2:
                    st.metric(
                        label=CURRENT_YEAR_COL_NAME,
                        value=f"{agg.iloc[-1]:,.0f}",
                        help=f"Aggregated value for {CURRENT_YEAR_COL_NAME}"
                    )

                with col3:
                    st.write("")  # Spacer for alignment
                    if st.session_state.aggregated_values and st.button(
                        "✅ Confirm New Values",
                        key=f"confirm_{item}"  # Unique key per tab
                    ):
                        for item, values in st.session_state.aggregated_values.items():
                            for col in [PREV_YEAR_COL_NAME, CURRENT_YEAR_COL_NAME]:
                                key = f"input_{item.lower()}_{col.lower().replace(' ', '_')}"
                                st.session_state[key] = str(values[col])
                        st.rerun()

                st.caption("Sum of: " + ", ".join(edited_df.loc[selected, 'label'].tolist()))

    st.html("<span class='big-dialog'></span>")


def create_data_option_toggles():
    # Add both toggles in a row
    col1, col2 = st.columns(2)
    with col1:
        st.toggle("Winsorize (before calculation)", key='winsorize',
                  help="Used to make the model more robust and less susceptible to the influence of extreme values")
    with col2:
        st.toggle("In millions (reloads data)", key='in_millions',
                  help="Display all values in millions (divided by 1,000,000)", on_change=on_ticker_change)


def create_input_form():
    """Create the main input form"""
    with st.form("financial_data_form"):
        st.subheader("Input data")
        st.write("Enter financial statement data for the latest financial reporting period and one year previously.")

        # # Add both toggles in a row
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.toggle("Winsorize (before calculation)", key='winsorize',
        #               help="Used to make the model more robust and less susceptible to the influence of extreme values")
        # with col2:
        #     st.toggle("In millions", key='in_millions',
        #               help="Display all values in millions (divided by 1,000,000)")

        # Four buttons used to calculate, clear, reload and hand-pick data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            submitted = st.form_submit_button("🚀 Calculate", help="Calculate the Beneish M-Score using the below accounting data")
        with col2:
            st.form_submit_button("🗑️ Clear data", on_click=clear_data, help="Both the ticker and any inputted accounting data will be cleared")
        with col3:
            st.form_submit_button("🔄️ Reload data", on_click=on_ticker_change, help="Reload data for original data for the specified ticker")
        with col4:
            if st.form_submit_button("⛏️ Hand-pick data", help="Opens a window where you can manually select which accounting items to include"):
                select_data_manually()  # This will open the dialog
        default_df = get_default_df()
        num_data_cols = len(default_df.columns)

        # Create header
        # cols = st.columns([1] + [1] * num_data_cols)
        cols = st.columns([1] + [1] * num_data_cols + [0.8])
        with cols[0]:
            st.write("")
        for i, col_name in enumerate(default_df.columns, 1):
            with cols[i]:
                st.markdown(f"**{col_name}**")
        with cols[-1]:  # New column for button header
            st.markdown("**Hand-pick item**")

        # Create input rows
        for index_item in default_df.index:
            # cols = st.columns([1] + [1] * num_data_cols)
            cols = st.columns([1] + [1] * num_data_cols + [0.8])
            with cols[0]:
                st.markdown(f"**{index_item}**", help=ACCOUNTING_ITEM_CONFIG[index_item]['help'])
            for i, col_name in enumerate(default_df.columns, 1):
                with cols[i]:
                    key = f"{index_item.lower()}_{col_name.lower().replace(' ', '_')}"
                    input_key = f"input_{key}"
                    st.text_input(
                        f"{index_item} {col_name}",
                        key=input_key,
                        label_visibility="collapsed"
                    )
            with cols[-1]:  # New column for buttons
                select_key = f"select_{index_item.lower().replace(' ', '_')}"
                st.checkbox(
                    "Select",
                    key=select_key,
                    label_visibility="collapsed"
                )
        return submitted, default_df


def check_variables(df: pd.DataFrame) -> tuple:
    """
    Checks so that all values are not negative and numeric
    Returns a tuple of (has_issues, message) where has_issues is boolean
    and message is a string describing the issues
    """
    issues = []

    # Check for null/missing values
    if df.isnull().values.any():
        issues.append("There are missing values in the input data.")

    # Check for negative values where they shouldn't exist
    for idx in get_non_negative_accounting_item_names():
        if (df.loc[idx] < 0).values.any():
            issues.append(f"Negative values found in {idx} which should be positive.")

    # Check for zero values where division might occur
    for idx in get_non_zero_accounting_item_names():
        if (df.loc[idx] == 0).any():
            issues.append(f"Zero values found in {idx} which may cause division errors in calculations.")

    if issues:
        return True, "WARNING: " + " ".join(issues)
    return False, "All input data appears valid."


def calculate_m_score_variables(df: pd.DataFrame, winsorize: bool) -> dict:
    """
    returns a dict with the 8 index variable used in the M score
    :param df: DataFrame
    :param winsorize: bool
    :return: dict
    """

    # Extract current year values
    curr_year_data = df[CURRENT_YEAR_COL_NAME]
    prev_year_data = df[PREV_YEAR_COL_NAME]

    result = {
        'DSRI': calculate_dsri(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'GMI': calculate_gmi(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'AQI': calculate_aqi(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'SGI': calculate_sgi(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'DEPI': calculate_depi(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'SGAI': calculate_sgai(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'LVGI': calculate_lvgi(curr_year_data=curr_year_data, prev_year_data=prev_year_data),
        'TATA': calculate_tata(curr_year_data=curr_year_data)
    }

    if winsorize:
        for k, v in result.copy().items():
            result[k] = max(min(WINSORIZE_VALUES[k]), min(max(WINSORIZE_VALUES[k]), v))

    return result


def calculate_dsri(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (curr_year_data[ACCOUNTS_RECEIVABLE_NAME] / curr_year_data[SALES_NET_NAME]) / \
        (prev_year_data[ACCOUNTS_RECEIVABLE_NAME] / prev_year_data[SALES_NET_NAME])


def calculate_gmi(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (prev_year_data[SALES_NET_NAME] - prev_year_data[COGS_NAME]) / prev_year_data[SALES_NET_NAME] / \
        ((curr_year_data[SALES_NET_NAME] - curr_year_data[COGS_NAME]) / curr_year_data[SALES_NET_NAME])


def calculate_aqi(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (1 - (curr_year_data[CURRENT_ASSETS_NAME] + curr_year_data[PPE_NET_NAME]) / curr_year_data[TOTAL_ASSET_NAME]) / \
        (1 - (prev_year_data[CURRENT_ASSETS_NAME] + prev_year_data[PPE_NET_NAME]) / prev_year_data[TOTAL_ASSET_NAME])


def calculate_sgi(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return curr_year_data[SALES_NET_NAME] / prev_year_data[SALES_NET_NAME]


def calculate_depi(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (prev_year_data[DEPRECIATION_AMORTIZATION_NAME] / (prev_year_data[PPE_NET_NAME] + prev_year_data[DEPRECIATION_AMORTIZATION_NAME])) / \
        (curr_year_data[DEPRECIATION_AMORTIZATION_NAME] / (curr_year_data[PPE_NET_NAME] + curr_year_data[DEPRECIATION_AMORTIZATION_NAME]))


def calculate_sgai(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (curr_year_data[SGA_EXPENSE_NAME] / curr_year_data[SALES_NET_NAME]) / \
        (prev_year_data[SGA_EXPENSE_NAME] / prev_year_data[SALES_NET_NAME])


def calculate_lvgi(curr_year_data: pd.Series, prev_year_data: pd.Series):
    return (curr_year_data[LONG_TERM_DEBT_NAME] + curr_year_data[CURRENT_LIABILITIES_NAME]) / curr_year_data[TOTAL_ASSET_NAME] / \
        ((prev_year_data[LONG_TERM_DEBT_NAME] + prev_year_data[CURRENT_LIABILITIES_NAME]) / prev_year_data[TOTAL_ASSET_NAME])


def calculate_tata(curr_year_data: pd.Series):
    return (curr_year_data[NET_INCOME_NAME] - curr_year_data[CFO_NAME]) / curr_year_data[TOTAL_ASSET_NAME]


def calculate_beneish_m_score(df: pd.DataFrame, winsorize: bool):
    """Calculate Beneish M-Score based on provided financial data"""
    try:
        variables = calculate_m_score_variables(df=df, winsorize=winsorize)
        # Calculate M-Score
        m_score = (-4.84 + 0.92 * variables['DSRI']
                   + 0.528 * variables['GMI']
                   + 0.404 * variables['AQI']
                   + 0.892 * variables['SGI']
                   + 0.115 * variables['DEPI']
                   - 0.172 * variables['SGAI']
                   + 4.679 * variables['TATA']
                   - 0.327 * variables['LVGI'])

        return m_score

    except (KeyError, ZeroDivisionError, TypeError):
        return None


def main():

    # Initialize session state
    initialize_session_state()

    # Title
    st.title("Beneish M-Score 📈")
    m_score_placeholder = st.empty()

    # SEC download section
    st.subheader("Download data from SEC for U.S. stocks")
    st.text_input("🔎 Enter ticker", placeholder="🔎 Enter ticker", key="ticker_input", on_change=on_ticker_change,
                  label_visibility='collapsed')

    # Calculation details (add a ✅ to the title M-score has been calculated)
    calculation_expander = st.expander("Calculation details", expanded=False)
    with calculation_expander:
        results_placeholder = st.empty()

    display_wiki()

    user_guide()

    create_data_option_toggles()

    # Create and handle form
    submitted, default_df = create_input_form()

    if submitted:
        try:
            # Collect submitted values
            data = {}
            for col_name in default_df.columns:
                col_values = []
                for index_item in default_df.index:
                    key = f"{index_item.lower()}_{col_name.lower().replace(' ', '_')}"
                    input_key = f"input_{key}"
                    value = st.session_state[input_key]
                    col_values.append(float(value) if value else np.nan)
                data[col_name] = col_values

            accounting_input_df = pd.DataFrame(data, index=default_df.index)
            has_issues, message = check_variables(accounting_input_df)

            if has_issues:
                st.warning(message)
            else:
                st.success("Data submitted successfully!")

            m_score = calculate_beneish_m_score(accounting_input_df, st.session_state.winsorize)
            if m_score is not None:
                variables = calculate_m_score_variables(df=accounting_input_df, winsorize=st.session_state.winsorize)

                # Display M-Score result
                with m_score_placeholder.container():
                    if has_issues:
                        st.warning("Calculation performed with data issues")

                    if m_score in [np.nan, np.inf, -np.inf]:
                        st.error(f"**⚠️ {m_score:.2f}** Not well defined score. Check 'calculation details'")
                    elif m_score > -1.78:
                        st.error(f"**🚨 {m_score:.2f}** Earnings might be manipulated")
                    elif -2 < m_score <= -1.78:
                        st.error(f"**🤔 {m_score:.2f}** Possible earnings manipulation")
                    else:
                        st.success(f"**🆗 {m_score:.2f}** Likely no manipulation")
                    st.markdown("> *Threshold: -1.78 (scores above may indicate manipulation)*")

                # Display calculation details
                with results_placeholder.container():
                    st.subheader("Beneish M-Score Components")
                    components = [
                        ("DSRI (Days Sales in Receivables Index)", variables['DSRI'], "0.920"),
                        ("GMI (Gross Margin Index)", variables['GMI'], "0.528"),
                        ("AQI (Asset Quality Index)", variables['AQI'], "0.404"),
                        ("SGI (Sales Growth Index)", variables['SGI'], "0.892"),
                        ("DEPI (Depreciation Index)", variables['DEPI'], "0.115"),
                        ("SGAI (SG&A Expense Index)", variables['SGAI'], "-0.172"),
                        ("LVGI (Leverage Index)", variables['LVGI'], "-0.327"),
                        ("TATA (Total Accruals to Total Assets)", variables['TATA'], "4.679")
                    ]

                    table = "| Component | Value | Weight |\n|-----------|-------|--------|\n"
                    for name, value, weight in components:
                        table += f"| {name} | {value:.4f} | {weight} |\n"
                    st.markdown(table)

                    st.markdown("**M-Score Formula:**")
                    st.latex(r'''
                    \begin{aligned}
                    M\text{-}Score &= -4.84 \\
                    &\quad + 0.920 \times DSRI \\
                    &\quad + 0.528 \times GMI \\
                    &\quad + 0.404 \times AQI \\
                    &\quad + 0.892 \times SGI \\
                    &\quad + 0.115 \times DEPI \\
                    &\quad - 0.172 \times SGAI \\
                    &\quad + 4.679 \times TATA \\
                    &\quad - 0.327 \times LVGI
                    \end{aligned}
                    ''')
            else:
                with m_score_placeholder.container():
                    st.warning("Could not calculate M-Score - please ensure all required fields are filled")

        except ValueError:
            st.error("Please enter valid numerical values")


if __name__ == '__main__':
    main()

