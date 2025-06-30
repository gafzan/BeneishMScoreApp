"""financial_statement_config.py"""

from pydantic import BaseModel
from typing import Union, List, Optional

from accounting_item_filter_config import REVENUE_ITEMS_FILTER, COGS_ITEMS_FILTER, DEPRECIATION_AMORTIZATION_FILTER, \
    SALES_GENERAL_ADMIN_FILTER, ACCOUNTS_RECEIVABLE_FILTER, CURRENT_ASSETS_FILTER, CURRENT_LIABILITIES_FILTER, \
    TOTAL_ASSETS_FILTER, PROPERTY_PLANT_EQUIPMENT_FILTER, LONG_TERM_DEBT_FILTER, NET_INCOME_FILTER, \
    CASH_FROM_OPERATIONS_FILTER, EPS_BASIC_FILTER, EPS_DILUTED_FILTER, NUM_SHARES_BASIC, NUM_SHARES_DILUTED
from accounting_item_filter_config import AccountingItemFilterConfig

# --- Synonyms ---
ANNUAL_SYNONYMS = {"annual", "annually", "12m", "12month", "12months", "twelvemonth", "twelvemonths", "10k", "10-k",
                   "yearly",
                   "years", "y"}
QUARTERLY_SYNONYMS = {'quarters', "quarterly", "quarter", "qrt", 'qtr', "q", "3month", "3months", "threemonth",
                      "threemonths", "3m",
                      "10q", "10-q"}
TTM_SYNONYMS = {"lasttwelvemonths", "lasttwelvemonth", 'ltm', "ttm", "t12m", "trailingtwelvemonth", "trailingtwelvemonths", "trailing12month", "trailing12months"}
INCOME_STATEMENT_SYNONYMS = {'income_statement', 'income', 'profit_and_loss', 'pnl', 'p&l', 'p&l_statement',
                             'pnl_statement', 'statement_of_income',
                             'is', 'consolidated_statement_of_income', 'consolidated_income_statement',
                             'statements_of_operations'}
BALANCE_SHEET_SYNONYMS = {'balance_sheet', 'bs', 'consolidated_balance_sheet'}
CASH_FLOW_STATEMENT_SYNONYMS = {'cash_flow_statement', 'cf', 'cash_flow', 'statement_of_cash_flow'}


# --- Constants ---
class AccountingItemKeys:
    REVENUES = 'revenues'
    COGS = 'cogs'
    CFO = 'cfo'
    DEPRECIATION_AMORTIZATION = 'depreciation_amortization'
    SALES_GENERAL_ADMIN = 'sales_general_admin'
    ACCOUNTS_RECEIVABLE = 'accounts_receivable'
    CURRENT_ASSETS = 'current_assets'
    CURRENT_LIABILITIES = 'current_liabilities'
    TOTAL_ASSETS = 'total_assets'
    PPE = 'ppe'
    LONG_TERM_DEBT = 'ltd'
    NET_INCOME = 'net_income'
    EPS_BASIC = 'eps_basic'
    EPS_DILUTED = 'eps_diluted'
    NUM_SHARES_BASIC = 'num_shares_basic'
    NUM_SHARES_DILUTED = 'num_shares_diluted'


# --- Schema ---
class FinancialItemConfig(BaseModel):
    statement: Union[str, List[str]]
    filter_config: AccountingItemFilterConfig
    filter_by_total: bool
    long_name: str
    instant: bool
    use_extended_data: Optional[bool] = None
    always_positive: bool

    def to_dict(self) -> dict:
        """Returns the filter as a dictionary, excluding unset fields"""
        return self.model_dump(exclude_unset=True)


def items_with_multiple_statements() -> list:
    """Returns the long name of accounting items that are included in more than one financial statement
    :return: list of str
    """
    return [config.long_name for config in FINANCIAL_STATEMENTS_CONFIG.values()
            if not isinstance(config.statement, str) and len(config.statement) > 1]


# --- Config ---
FINANCIAL_STATEMENTS_CONFIG = {
    AccountingItemKeys.REVENUES: FinancialItemConfig(
        statement='income_statement',
        filter_config=REVENUE_ITEMS_FILTER,
        filter_by_total=True,
        long_name='Revenues',
        instant=False
        ,use_extended_data=False
        ,always_positive=True
),
    AccountingItemKeys.COGS: FinancialItemConfig(
        statement='income_statement',
        filter_config=COGS_ITEMS_FILTER,
        filter_by_total=True,
        long_name='Cost of Goods Sold',
        instant=False
        , always_positive=True
    ),
    AccountingItemKeys.CFO: FinancialItemConfig(
        statement='cashflow_statement',
        filter_config=CASH_FROM_OPERATIONS_FILTER,
        filter_by_total=True,
        long_name='Cash from Operations',
        instant=False
        , always_positive=False
    ),
    AccountingItemKeys.DEPRECIATION_AMORTIZATION: FinancialItemConfig(
        statement='cashflow_statement',
        filter_config=DEPRECIATION_AMORTIZATION_FILTER,
        filter_by_total=False,
        long_name='Depreciation & Amortization Expense',
        instant=False
        , always_positive=True
    ),
    AccountingItemKeys.SALES_GENERAL_ADMIN: FinancialItemConfig(
        statement='income_statement',
        filter_config=SALES_GENERAL_ADMIN_FILTER,
        filter_by_total=False,
        long_name='Selling, General, and Administrative Expenses (SG&A)',
        instant=False
        , always_positive=True
    ),
    AccountingItemKeys.ACCOUNTS_RECEIVABLE: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=ACCOUNTS_RECEIVABLE_FILTER,
        filter_by_total=False,
        long_name='Accounts Receivable',
        instant=True
        , always_positive=True
    ),
    AccountingItemKeys.CURRENT_ASSETS: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=CURRENT_ASSETS_FILTER,
        filter_by_total=True,
        long_name='Current Asset',
        instant=True
        , always_positive=True
    ),
    AccountingItemKeys.CURRENT_LIABILITIES: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=CURRENT_LIABILITIES_FILTER,
        filter_by_total=True,
        long_name='Current Liabilities',
        instant=True
        , always_positive=True
    ),
    AccountingItemKeys.TOTAL_ASSETS: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=TOTAL_ASSETS_FILTER,
        filter_by_total=True,
        long_name='Total Assets',
        instant=True
        ,use_extended_data=False
        , always_positive=True
    ),
    AccountingItemKeys.PPE: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=PROPERTY_PLANT_EQUIPMENT_FILTER,
        filter_by_total=False,
        long_name='Property, Plant & Equipment (PPE)',
        instant=True
        , always_positive=True
    ),
    AccountingItemKeys.LONG_TERM_DEBT: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=LONG_TERM_DEBT_FILTER,
        filter_by_total=False,
        long_name='Long Term Debt',
        instant=True
        , always_positive=True
    ),
    AccountingItemKeys.NET_INCOME: FinancialItemConfig(
        statement='income_statement',
        filter_config=NET_INCOME_FILTER,
        filter_by_total=False,
        long_name='Net Income',
        instant=False
        , use_extended_data=False
        , always_positive=False
    ),
    AccountingItemKeys.EPS_BASIC: FinancialItemConfig(
        statement='income_statement',
        filter_config=EPS_BASIC_FILTER,
        filter_by_total=False,
        long_name='EPS (Basic)',
        instant=False
        , use_extended_data=False
        , always_positive=False
    ),
    AccountingItemKeys.EPS_DILUTED: FinancialItemConfig(
        statement='income_statement',
        filter_config=EPS_DILUTED_FILTER,
        filter_by_total=False,
        long_name='EPS (Diluted)',
        instant=False
        , use_extended_data=False
        , always_positive=False
    ),
    AccountingItemKeys.NUM_SHARES_BASIC: FinancialItemConfig(
        statement='income_statement',
        filter_config=NUM_SHARES_BASIC,
        filter_by_total=False,
        long_name='Number of Shares (Basic)',
        instant=True
        , use_extended_data=False
        , always_positive=True
    ),
    AccountingItemKeys.NUM_SHARES_DILUTED: FinancialItemConfig(
        statement='income_statement',
        filter_config=NUM_SHARES_DILUTED,
        filter_by_total=False,
        long_name='Number of Shares (Diluted)',
        instant=True
        , use_extended_data=False
        , always_positive=True
    ),
}
