import os
import pandas as pd
from fredapi import Fred

apikey = "4fcd34055360a6293e3f9f66e75ea562"
fred = Fred(api_key=apikey)

def search_series(query:str) -> dict:
    return list(fred.search(query, limit=1, order_by="search_rank", sort_order="desc").T.to_dict().values())[0]

def get_fred_series(series_id: str, start_date: str = '2020-01-01') -> pd.DataFrame:
    series_data = fred.get_series(series_id, observation_start=start_date)
    series_data.name = series_id

    if series_data.empty:
        return pd.DataFrame()
    series_data.ffill(inplace=True)
    series_data.bfill(inplace=True)
    series_data.dropna(how='all', inplace=True)

    return series_data

def get_fred_series_as_observed(series_id: str) -> pd.DataFrame:
    series_data = fred.get_series_first_release(series_id)
    series_data.name = series_id 

    if series_data.empty:
        return pd.DataFrame()
    series_data.ffill(inplace=True)
    series_data.bfill(inplace=True)
    series_data.dropna(how='all', inplace=True)

    return series_data