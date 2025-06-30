"""accounting_item_filter_config.py"""
import json

from pydantic import BaseModel, ValidationError, Field
from typing import Optional, List, Dict, Any, Tuple, Union


class AccountingItemFilterConfig(BaseModel):
    """
    Class describing how each config needs to be defined

    Each field is marked as Optional, meaning it can be omitted or set to None.
    """
    exact_concepts: Optional[List[str]] = Field(
        None,
        description="Filter items that exactly match these concepts ignoring common suffixes like 'us-gaap_' or capital "
                    "letters (e.g., ['RevenueFromContractWithCustomer', 'Revenues'])."
    )
    exact_labels: Optional[List[str]] = Field(
        None,
        description="Filter items that exactly match these labels ignoring capital letters (e.g., ['Total Revenues', 'Net Sales'])."
    )
    single_concept_sequence: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Starting with the first concept filtering element (could be a str or iterable of str), if the "
                    "result contains one row, result is returned, else go to the next one until the end of the list."
                    "E.g. [('netincome', 'common'), 'netincome'] will first try if rows exists where concept includes "
                    "'netincome' AND 'common' as sub strings. If result is has more than one row or empty, go to the "
                    "next one wich is only 'netincome'"
    )
    single_label_sequence: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Starting with the first label filtering element (could be a str or iterable of str), if the "
                    "result contains one row, result is returned, else go to the next one until the end of the list."
                    "E.g. [('net income', 'common'), 'netincome'] will first try if rows exists where label includes "
                    "'net income' AND 'common' as sub strings. If result is has more than one row or empty, go to the "
                    "next one wich is only 'net income'"
    )
    non_empty_concept_sequence: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Starting with the first concept filtering element (could be a str or iterable of str), if the "
                    "result is non-empty, result is returned, else go to the next one until the end of the list."
                    "E.g. [('netincome', 'common'), 'netincome'] will first try if rows exists where concept includes "
                    "'netincome' AND 'common' as sub strings. If result is empty, go to the next one wich is only "
                    "'netincome'"
    )
    non_empty_label_sequence: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Starting with the first label filtering element (could be a str or iterable of str), if the "
                    "result is non-empty, result is returned, else go to the next one until the end of the list."
                    "E.g. [('net income', 'common'), 'net income'] will first try if rows exists where label includes "
                    "'net income' AND 'common' as sub strings. If result is empty, go to the next one which is only "
                    "'net income'"
    )
    partial_concepts: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Filter items that partially match these concepts ignoring capital letters (e.g., ['costof', 'cogs', 'expense'])."
                    "Can also have iterables INSIDE the list that implies AND logic i.e. ALL substrings needs to be present in concept for it to be filtered."
                    "E.g. 'partial_concepts': [('current', 'asset')] filters concepts that has BOTH 'current' and 'asset' as substrings."
    )
    partial_labels: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Filter items that partially match these labels ignoring capital letters (e.g., ['costof', 'cogs', 'expense'])."
                    "Can also have iterables INSIDE the list that implies AND logic i.e. ALL substrings needs to be present in labels for it to be filtered."
                    "E.g. 'partial_labels': [('current', 'asset')] filters concepts that has BOTH 'current' and 'asset' as substrings."
    )
    partial_exclusions: Optional[List[Union[str, List[str], Tuple[str]]]] = Field(
        None,
        description="Does not filter items that has any of these substrings (e.g. ['fee', 'tax', 'operating'])"
    )
    concept_label_union: Optional[bool] = Field(
        None,
        description="If True, matches for concept and label as a union"
    )

    @classmethod
    def print_input_structure(cls):
        schema = cls.describe_input_structure()
        print(json.dumps(schema, indent=2))

    def to_dict(self) -> dict:
        """Returns the filter as a dictionary, excluding unset fields"""
        return self.model_dump(exclude_unset=True)

    @classmethod
    def describe_input_structure(cls) -> Dict[str, Dict[str, Any]]:
        schema = cls.model_json_schema()
        return {
            field_name: {
                "type": field_info.get("type"),
                "required": field_info.get("required", False),
                "default": field_info.get("default"),
                "description": field_info.get("description")
            }
            for field_name, field_info in schema["properties"].items()
        }

    @classmethod
    def validate_input_structure(cls, filter_config: dict):
        try:
            cls(**filter_config).model_dump(exclude_unset=True)
        except ValidationError as e:
            raise ValueError(f"Invalid input: {e}")

# ----------------------------------------------------------------------------------------------------------------------
# Define new filter configurations below


# Revenue Items
REVENUE_ITEMS_FILTER = AccountingItemFilterConfig(
    exact_concepts=[
        'Revenue',
        'Revenues',
        'RevenueFromContractWithCustomer',
        'RevenueFromProductSales',
        'RevenueFromContractWithCustomerExcludingAssessedTax'
    ],
    exact_labels=['Total Revenues', 'Revenue', 'Net sales'],
    partial_concepts=['revenue', 'sales'],
    partial_labels=['revenue', 'sales', 'income'],
    partial_exclusions=['cost', 'cogs', 'other'],
)

# Cost of Goods Sold
COGS_ITEMS_FILTER = AccountingItemFilterConfig(
    exact_concepts=['CostOfRevenue', 'CostOfGoodsSold'],
    partial_concepts=['costof', 'cogs', 'expense', 'purchase'],
    partial_labels=['cost of goods', 'cogs', 'expense', 'purchase'],
    partial_exclusions=[
        'interest', 'operating', 'depreciation', 'amortization',
        'marketing', 'general', 'admin', 'stock', 'write',
        'occurring', 'tax', 'research', 'development', 'r&d', 'lease', 'pension', 'retirement'
    ]
)

# Depreciation & Amortization
DEPRECIATION_AMORTIZATION_FILTER = AccountingItemFilterConfig(
    exact_concepts=['DepreciationDepletionAndAmortization'],
    exact_labels=[
        'depreciation and amortization',
        'depreciation & amortization'
    ],
    partial_concepts=['depreciation', 'amortization'],
    partial_labels=['depreciation', 'amortization'],
    partial_exclusions=[]
)

# SG&A Expenses
SALES_GENERAL_ADMIN_FILTER = AccountingItemFilterConfig(
    partial_concepts=['marketing', 'general', 'admin'],
    partial_labels=[
        'marketing',
        'administrative',
        ('general', 'admin'),
        ('sale', 'general'),
        'sga',
        'sg&a'
    ],
    partial_exclusions=['fee', 'tax', 'operating']
)

# Accounts Receivable
ACCOUNTS_RECEIVABLE_FILTER = AccountingItemFilterConfig(
    exact_concepts=['ReceivablesNetCurrent'],
    exact_labels=[
        'Accounts receivable',
        'account receivable',
        'receivables',
        'current receivables'
    ],
    partial_concepts=['receivable'],
    partial_labels=['receivable']
)

# Current Assets
CURRENT_ASSETS_FILTER = AccountingItemFilterConfig(
    exact_concepts=['AssetsCurrent'],
    exact_labels=['current assets', 'total current assets'],
    partial_concepts=[('current', 'asset')],
    partial_labels=[('current', 'asset')],
    partial_exclusions=['other']
)

# Current Liabilities
CURRENT_LIABILITIES_FILTER = AccountingItemFilterConfig(
    exact_concepts=['LiabilitiesCurrent'],
    exact_labels=['current liabilities', 'total current liabilities'],
    partial_concepts=[('current', 'liabilit')],
    partial_labels=[('current', 'liabilit')],
    partial_exclusions=['other']
)

# Total Assets
TOTAL_ASSETS_FILTER = AccountingItemFilterConfig(
    exact_concepts=['Assets', 'Asset'],
    exact_labels=['total asset', 'total assets'],
    partial_concepts=[('total', 'asset')],
    partial_labels=[('total', 'asset')],
    partial_exclusions=['current']
)

# PPE (Property, Plant & Equipment)
PROPERTY_PLANT_EQUIPMENT_FILTER = AccountingItemFilterConfig(
    partial_concepts=[
        ('property', 'plant', 'equipment'),
        ('property', 'equipment'),
        ('right', 'use', 'asset'),
        ('lease', 'asset'),
        ('leasing', 'asset'),
        ('deferredcost', 'leasing', 'noncurrent'),
        ('deferredcost', 'lease', 'noncurrent'),
    ],
    partial_labels=[
        ('property', 'plant', 'equipment'),
        ('property', 'equipment'),
        ('right', 'use', 'asset'),
        ('lease', 'asset'),
        ('leasing', 'asset')
    ],
    partial_exclusions=['liabilit'],
    concept_label_union=True
)

# Long Term Debt
LONG_TERM_DEBT_FILTER = AccountingItemFilterConfig(
    exact_concepts=['LongTermDebtAndCapitalLeaseObligations', 'LongTermDebtAndFinanceLeasesNoncurrent'],
    partial_concepts=[
        ('long', 'term', 'debt'),
        ('long', 'term', 'borrowing'),
        ('non', 'current', 'debt'),
        ('non', 'current', 'borrowing'),
        ('lease', 'non', 'current'),
        ('lease', 'long', 'term')
    ],
    partial_labels=[
        ('long', 'term', 'debt'),
        ('long', 'term', 'borrowing'),
        ('non', 'current', 'debt'),
        ('non', 'current', 'borrowing'),
        ('lease', 'non', 'current'),
        ('lease', 'long', 'term')
    ],
    partial_exclusions=['LongTermDebtCurrent']
)

# Net Income
NET_INCOME_FILTER = AccountingItemFilterConfig(
    # exact_concepts=[
    #     'ProfitLoss',
    #     'IncomeLossFromContinuingOperationsNetOfTaxAvailableToCommonStockholders',
    #     'NetIncomeLoss'
    # ],
    # partial_concepts=['ProfitLoss', 'netincome'],
    # partial_labels=['net income'],
    # ('earnings', 'common'), ('earnings', 'company'), 'earnings'
    # exact_labels=['net loss'],
    non_empty_label_sequence=[('net income', 'common'), 'net income', ('earnings', 'common'), ('earnings', 'company'), 'earnings'],
    partial_concepts=['netincomeloss'],
    partial_exclusions=['minority', 'discontinued', 'per share', 'pershare', 'before', 'controlling']
)

# Cash from Operations
CASH_FROM_OPERATIONS_FILTER = AccountingItemFilterConfig(
    exact_concepts=['NetCashProvidedByUsedInOperatingActivities'],
    exact_labels=['Net Cash from Operations'],
    partial_concepts=[('Cash', 'Operat')],
    partial_labels=[('Cash', 'Operat')],
    partial_exclusions=['per share', 'pershare', 'other']
)

# EPS basic
EPS_BASIC_FILTER = AccountingItemFilterConfig(
    exact_concepts=[
        'EarningsPerShareBasic',
    ],
    partial_concepts=[('per', 'basic', 'share')],
    # partial_exclusions=['discontinued', 'per share', 'pershare']
)

# EPS basic
EPS_DILUTED_FILTER = AccountingItemFilterConfig(
    exact_concepts=[
        'EarningsPerShareDiluted',
    ],
    partial_concepts=[('per', 'diluted', 'share')],
    # partial_exclusions=['discontinued', 'per share', 'pershare']
)

# Number of shares basic
NUM_SHARES_BASIC = AccountingItemFilterConfig(
    exact_concepts=[
        'WeightedAverageNumberOfSharesOutstandingBasic',
    ],
    partial_concepts=[('number', 'shares', 'basic')],
    # partial_exclusions=['discontinued', 'per share', 'pershare']
)

# Number of shares diluted
NUM_SHARES_DILUTED = AccountingItemFilterConfig(
    exact_concepts=[
        'WeightedAverageNumberOfDilutedSharesOutstanding',
    ],
    partial_concepts=[('number', 'shares', 'diluted')],
    # partial_exclusions=['discontinued', 'per share', 'pershare']
)
