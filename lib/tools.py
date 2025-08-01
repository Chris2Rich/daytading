from retrieval.filerag import rag_search, Chunk, metadata
import retrieval.yahoofinance as yfwrapper
import retrieval.fred as fredwrapper
import modelling.fundamental as fundamentalwrapper
import pandas as pd
import quantstats as qs
import numpy as np
import datetime

context = None

def tool_calculate_wacc(equity_market_value: float, debt_market_value: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> float:
    """
    Use this to calculate the Weighted Average Cost Of Capital
    """
    return fundamentalwrapper.wacc(equity_market_value, debt_market_value, cost_of_equity, cost_of_debt, tax_rate)

def tool_calculate_dcf(expected_cashflow: list, wacc: list, initial_investment: float) -> float:
    """
    Use this to calculate the net present value of a cashflow producing asset by applying projected WACCs as discount factors over projected cashflows
    """
    expected_cashflow_array = np.array(expected_cashflow)
    wacc_array = np.array(wacc)
    return fundamentalwrapper.dcf(expected_cashflow_array, wacc_array, initial_investment)

def tool_calculate_pe_ratio(price: float, earnings: float) -> float:
    """
    Use this to calculate the price to earnings ratio (market value signal)
    """
    return fundamentalwrapper.pe_ratio(price, earnings)

def tool_calculate_debt_to_equity(total_debt: float, total_equity: float) -> float:
    """
    Use this to calculate debt to equity ratio of a company (leverage signal)
    """
    return fundamentalwrapper.debt_to_equity(total_debt, total_equity)

def tool_calculate_roe(net_income: float, shareholders_equity: float) -> float:
    """
    Use this to calculate the return on equity of a company (efficiency of useage of equity)
    """
    return fundamentalwrapper.roe(net_income, shareholders_equity)

def tool_calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Use this to calculate the current ratio of a company (short term liquidity signal)
    """
    return fundamentalwrapper.current_ratio(current_assets, current_liabilities)

def tool_calculate_debt_service_coverage(net_operating_income: float, debt_service: float) -> float:
    """
    Use this to calculate debt service coverage (ability to pay cost of total debt)
    """
    return fundamentalwrapper.debt_service_coverage(net_operating_income, debt_service)

def tool_calculate_interest_coverage(ebit: float, interest_expense: float) -> float:
    """
    Use this to calculate interest coverage (ability to pay cost of borrowing)
    """
    return fundamentalwrapper.interest_coverage(ebit, interest_expense)

def tool_query_filing_data(query: str) -> str:
    """
    For retrieving textual factual reported data from the past:
    Use this tool to find and extract specific textual information directly from a company's current (8-K), annual (10-K) and quarterly (10-Q) SEC filings. It is best for answering questions about reported business risks, management's discussion and analysis (MD&A), and executive compensation. The input should be a clear, specific question about the company's filings. Returns the most relevant text snippets from the documents. This can also be used to gather information about long term company strategy and management's explanations for short-long term trends in financial fundamentals.

    Do NOT use this for hypothetical or "what-if" scenarios.
    """
    return "\n".join([f"<retrieval {index}>" + summary["summary"] + f"</retrieval {index}>" for index, summary in enumerate(rag_search(context=context, search_q=query, should_summarize=False))])

def tool_get_date_time() -> str:
    """
    For getting the current date and time (in UTC time) as ISO format string
    """
    return datetime.datetime.utcnow().isoformat()

def tool_get_trading_recent(ticker: str, days: int, interval: str) -> dict:
    """
    For getting the last "days" days of trading data for a stock with ticker "ticker" with interval "interval" (time of each tick)
    Intervals are:
        "1mo" - monthly (good for comparing against macroeconomic timeseries)
        "1d" - daily
        "1h" - hourly
        "30m"
        "15m"
        "5m"
        "1m" - minutely
        minutely and hourly data is only available for the last 7 days
    """
    df_data = yfwrapper.get_trading(ticker, days, interval)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_get_option_chain(ticker: str, expiry: str = None) -> dict:
    """
    Get the current option chain for a ticker.
    """
    df_data = yfwrapper.get_option_chain(ticker, expiry)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_get_company_financials(ticker: str, statement_type: str = 'income_statement', period: str = 'annual') -> dict:
    """
    Retrieves a company's financial statements (income_statement, balance_sheet, or cash_flow).
    Specify 'annual' or 'quarterly' for the period.
    """
    df_data = yfwrapper.get_fundamentals(ticker, statement_type, period)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_get_FRED_macro_ids(query: str) -> dict:
    """
    Retrieves a dictionary containing id and metadata for the most relevant macroeconomic timeseries in respect to the search query. Ids are for use with FRED tools. The input should be a clear measurable concept.
    """
    df_data = fredwrapper.search_series(query)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_get_FRED_macro(series_id: str, start_date: str = "2020-01-01") -> dict:
    """
    Retrieves a dictionary containing the relevant macroeconomic data as a timeseries based on the series_id. Start date is the provided start date (default is Jan 1 2020).
    """
    df_data = fredwrapper.get_fred_series(series_id, start_date)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_get_FRED_macro_as_observed(series_id: str) -> dict:
    """
    Retrieves a dictionary containing the relevant macroeconomic data as it was observed at the time (excludes revisions) as a timeseries based on the series_id. This can be useful in making models as inaccuracies of the time are considered.
    """
    df_data = fredwrapper.get_fred_series_as_observed(series_id)
    if isinstance(df_data, pd.DataFrame):
        return df_data.to_dict(orient='records')
    return df_data

def tool_analyze_strategy_returns(returns: list) -> dict:
    """
    Takes in returns of a strategy (as list) and returns sharpe ratio, win rate, max drawdown and cagr 
    """
    # Convert list to pandas Series for quantstats
    returns_series = pd.Series(returns)
    
    stats = {}
    stats['sharpe'] = float(qs.stats.sharpe(returns_series))
    stats['max_drawdown'] = float(qs.stats.max_drawdown(returns_series))
    stats['cagr'] = float(qs.stats.cagr(returns_series))
    stats['win_rate'] = float(qs.stats.win_rate(returns_series))
    
    return stats