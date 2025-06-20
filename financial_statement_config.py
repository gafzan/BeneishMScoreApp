"""financial_statement_config.py"""

from pydantic import BaseModel
from typing import Union, List

from accounting_item_filter_config import REVENUE_ITEMS_FILTER, COGS_ITEMS_FILTER, DEPRECIATION_AMORTIZATION_FILTER, \
    SALES_GENERAL_ADMIN_FILTER, ACCOUNTS_RECEIVABLE_FILTER, CURRENT_ASSETS_FILTER, CURRENT_LIABILITIES_FILTER, \
    TOTAL_ASSETS_FILTER, PROPERTY_PLANT_EQUIPMENT_FILTER, LONG_TERM_DEBT_FILTER, NET_INCOME_FILTER, \
    CASH_FROM_OPERATIONS_FILTER
from accounting_item_filter_config import AccountingItemFilterConfig


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


# --- Schema ---
class FinancialItemConfig(BaseModel):
    statement: Union[str, List[str]]
    filter_config: AccountingItemFilterConfig
    filter_by_total: bool
    long_name: str
    instant: bool


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
    ),
    AccountingItemKeys.COGS: FinancialItemConfig(
        statement='income_statement',
        filter_config=COGS_ITEMS_FILTER,
        filter_by_total=True,
        long_name='Cost of Goods Sold',
        instant=False
    ),
    AccountingItemKeys.CFO: FinancialItemConfig(
        statement='cashflow_statement',
        filter_config=CASH_FROM_OPERATIONS_FILTER,
        filter_by_total=True,
        long_name='Cash from Operations',
        instant=False
    ),
    AccountingItemKeys.DEPRECIATION_AMORTIZATION: FinancialItemConfig(
        statement=['income_statement', 'cashflow_statement'],
        filter_config=DEPRECIATION_AMORTIZATION_FILTER,
        filter_by_total=False,
        long_name='Depreciation & Amortization Expense',
        instant=False
    ),
    AccountingItemKeys.SALES_GENERAL_ADMIN: FinancialItemConfig(
        statement='income_statement',
        filter_config=SALES_GENERAL_ADMIN_FILTER,
        filter_by_total=False,
        long_name='Selling, General, and Administrative Expenses (SG&A)',
        instant=False
    ),
    AccountingItemKeys.ACCOUNTS_RECEIVABLE: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=ACCOUNTS_RECEIVABLE_FILTER,
        filter_by_total=False,
        long_name='Accounts Receivable',
        instant=True
    ),
    AccountingItemKeys.CURRENT_ASSETS: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=CURRENT_ASSETS_FILTER,
        filter_by_total=False,
        long_name='Current Asset',
        instant=True
    ),
    AccountingItemKeys.CURRENT_LIABILITIES: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=CURRENT_LIABILITIES_FILTER,
        filter_by_total=False,
        long_name='Current Liabilities',
        instant=True
    ),
    AccountingItemKeys.TOTAL_ASSETS: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=TOTAL_ASSETS_FILTER,
        filter_by_total=False,
        long_name='Total Assets',
        instant=True
    ),
    AccountingItemKeys.PPE: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=PROPERTY_PLANT_EQUIPMENT_FILTER,
        filter_by_total=False,
        long_name='Property, Plant & Equipment (PPE)',
        instant=True
    ),
    AccountingItemKeys.LONG_TERM_DEBT: FinancialItemConfig(
        statement='balance_sheet',
        filter_config=LONG_TERM_DEBT_FILTER,
        filter_by_total=False,
        long_name='Long Term Debt',
        instant=True
    ),
    AccountingItemKeys.NET_INCOME: FinancialItemConfig(
        statement=['cashflow_statement', 'income_statement'],
        filter_config=NET_INCOME_FILTER,
        filter_by_total=False,
        long_name='Net Income',
        instant=True
    ),
}
