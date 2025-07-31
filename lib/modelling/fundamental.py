import numpy as np 

def wacc(equity_market_value: float, debt_market_value: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> float:
    return ((equity_market_value * cost_of_equity) + (debt_market_value * cost_of_debt * (1- tax_rate))) / (equity_market_value + debt_market_value)

def dcf(expected_cashflow: np.array, wacc: np.array, initial_investment: float) -> float:
    return np.sum(expected_cashflow / np.cumprod(wacc + 1)) - initial_investment

def pe_ratio(price: float, earnings_per_share: float) -> float:
    return price / earnings_per_share

def debt_to_equity(total_debt: float, total_equity: float) -> float:
    return total_debt / total_equity

def roe(net_income: float, shareholders_equity: float) -> float:
    return net_income / shareholders_equity

def current_ratio(current_assets: float, current_liabilities: float) -> float:
    return current_assets / current_liabilities

def debt_service_coverage(net_operating_income: float, debt_service: float) -> float:
    return net_operating_income / debt_service

def interest_coverage(ebit: float, interest_expense: float) -> float:
    return ebit / interest_expense