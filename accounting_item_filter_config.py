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
    partial_exclusions=['cost', 'cogs', 'other']
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
    partial_labels=[('current', 'asset')]
)

# Current Liabilities
CURRENT_LIABILITIES_FILTER = AccountingItemFilterConfig(
    exact_concepts=['LiabilitiesCurrent'],
    exact_labels=['current liabilities', 'total current liabilities'],
    partial_concepts=[('current', 'liabilit')],
    partial_labels=[('current', 'liabilit')]
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
        ('right', 'use', 'asset')
    ],
    partial_labels=[
        ('property', 'plant', 'equipment'),
        ('property', 'equipment'),
        ('right', 'use', 'asset')
    ],
    partial_exclusions=['liabilit', 'current']
)

# Long Term Debt
LONG_TERM_DEBT_FILTER = AccountingItemFilterConfig(
    exact_concepts=['LongTermDebtAndCapitalLeaseObligations'],
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
    partial_exclusions=[]
)

# Net Income
NET_INCOME_FILTER = AccountingItemFilterConfig(
    exact_concepts=[
        'ProfitLoss',
        'IncomeLossFromContinuingOperationsNetOfTaxAvailableToCommonStockholders',
        'NetIncomeLoss'
    ],
    partial_concepts=['ProfitLoss', 'netincome'],
    partial_labels=['net income'],
    partial_exclusions=['discontinued', 'per share', 'pershare']
)

# Cash from Operations
CASH_FROM_OPERATIONS_FILTER = AccountingItemFilterConfig(
    exact_concepts=['NetCashProvidedByUsedInOperatingActivities'],
    exact_labels=['Net Cash from Operations'],
    partial_concepts=[('Cash', 'Operat')],
    partial_labels=[('Cash', 'Operat')],
    partial_exclusions=['per share', 'pershare']
)

