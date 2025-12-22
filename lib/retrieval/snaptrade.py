import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from clerk_backend_api import Clerk
import math

# Import SnapTrade SDK
try:
    from snaptrade_client import SnapTrade
except ImportError:
    print("ERROR: snaptrade-python-sdk not installed. Run: pip install snaptrade-python-sdk")
    SnapTrade = None

# SnapTrade configuration
SNAPTRADE_CLIENT_ID = os.environ.get("SNAPTRADE_CLIENT_ID")
SNAPTRADE_SECRET = os.environ.get("SNAPTRADE_SECRET")

if not SNAPTRADE_CLIENT_ID or not SNAPTRADE_SECRET:
    print("Warning: SnapTrade credentials not configured")

# Initialize Clerk SDK
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)

def safe_float(value) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0

def format_trades(activities: list) -> list:
    """Filter and format trade activities"""
    if not isinstance(activities, list):
        return []
    
    trades = []
    for activity in activities:
        activity_type = (activity.get("type") or "").lower()
        if "buy" in activity_type or "sell" in activity_type or "trade" in activity_type:
            symbol_obj = activity.get("symbol", {})
            symbol = symbol_obj.get("symbol") if isinstance(symbol_obj, dict) else symbol_obj
            
            trades.append({
                "date": activity.get("trade_date") or activity.get("settlement_date") or activity.get("date"),
                "timestamp": activity.get("trade_date") or activity.get("settlement_date"),
                "side": activity.get("type"),
                "type": activity.get("type"),
                "symbol": symbol,
                "ticker": symbol,
                "quantity": abs(activity.get("units", 0) or activity.get("quantity", 0)),
                "qty": abs(activity.get("units", 0) or activity.get("quantity", 0)),
                "price": activity.get("price", 0),
                "amount": activity.get("amount", 0),
                "fee": activity.get("fee", 0),
                "description": activity.get("description"),
            })
    
    # Sort by date descending
    trades.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return trades


async def fetch_portfolio_from_snaptrade(clerk_user_id: str) -> Dict[str, Any]:
    """
    Fetch portfolio data from SnapTrade API using the official SDK.
    """
    if not SNAPTRADE_CLIENT_ID or not SNAPTRADE_SECRET:
        return {
            "connected": False,
            "error": "SnapTrade not configured",
            "accounts": [],
            "positions": [],
            "trades": [],
            "totalNav": 0.0
        }
    
    if not SnapTrade:
        return {
            "connected": False,
            "error": "SnapTrade SDK not installed",
            "accounts": [],
            "positions": [],
            "trades": [],
            "totalNav": 0.0
        }
    
    try:
        # Get SnapTrade credentials from Clerk
        user_response = clerk.users.get(user_id=clerk_user_id)
        private_metadata = user_response.private_metadata or {}
        
        snaptrade_user_id = private_metadata.get("snaptradeUserId", clerk_user_id)
        snaptrade_user_secret = private_metadata.get("snaptradeUserSecret")
        
        if not snaptrade_user_secret:
            return {
                "connected": False,
                "error": "SnapTrade not connected",
                "accounts": [],
                "positions": [],
                "trades": [],
                "totalNav": 0.0
            }
        
        # Initialize SnapTrade SDK client
        snaptrade = SnapTrade(
            client_id=SNAPTRADE_CLIENT_ID,
            consumer_key=SNAPTRADE_SECRET
        )
        
        # Fetch accounts
        accounts_resp = snaptrade.account_information.list_user_accounts(
            user_id=snaptrade_user_id,
            user_secret=snaptrade_user_secret
        )
        
        # Extract accounts from response
        accounts = accounts_resp.body if hasattr(accounts_resp, 'body') else accounts_resp
        if not accounts:
            return {
                "connected": True,
                "accounts": [],
                "positions": [],
                "trades": [],
                "totalNav": 0.0
            }
        
        # Fetch positions and activities for each account
        all_positions = []
        all_activities = []
        account_ids = [acc["id"] for acc in accounts]
        accounts_param = ",".join(account_ids)

        activities_resp = snaptrade.transactions_and_reporting.get_activities(
            user_id=snaptrade_user_id,
            user_secret=snaptrade_user_secret,
            accounts=accounts_param,
        )

        activities_data = activities_resp.body if hasattr(activities_resp, 'body') else activities_resp
        if isinstance(activities_data, list):
            all_activities.extend(activities_data)
        elif isinstance(activities_data, dict) and "data" in activities_data:
            all_activities.extend(activities_data["data"])

        for account in accounts:
            account_id = account.get("id")
            
            try:
                # Get positions
                positions_resp = snaptrade.account_information.get_user_account_positions(
                    user_id=snaptrade_user_id,
                    user_secret=snaptrade_user_secret,
                    account_id=account_id
                )
                positions = positions_resp.body if hasattr(positions_resp, 'body') else positions_resp
                if positions:
                    all_positions.extend(positions)
                    
            except Exception as e:
                print(f"Error fetching data for account {account_id}: {e}")
                continue
        
        # Process data
        totalNav = sum(
            safe_float(acc.get("balance", {}).get("total"))
            for acc in accounts
        )
        trades = format_trades(all_activities)
        
        # Format accounts
        formatted_accounts = [
            {
                "id": acc.get("id"),
                "name": acc.get("name"),
                "accountNumber": acc.get("number"),
                "brokerage": acc.get("institution_name") or acc.get("brokerage_name"),
                "type": acc.get("type"),
            }
            for acc in accounts
        ]
        
        return {
            "connected": True,
            "accounts": formatted_accounts,
            "positions": all_positions,
            "trades": trades,
            "totalNav": totalNav
        }
    
    except Exception as e:
        print(f"Error fetching portfolio from SnapTrade: {e}")
        import traceback
        traceback.print_exc()
        return {
            "connected": False,
            "error": str(e),
            "accounts": [],
            "positions": [],
            "trades": [],
            "totalNav": 0.0
        }


async def get_portfolio_data(
    clerk_user_id: str,
    db,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Get portfolio data with intelligent caching."""
    cache_max_age_minutes = 20
    
    if not force_refresh:
        try:
            cached = await db.get_portfolio_cache(clerk_user_id, max_age_minutes=cache_max_age_minutes)
            
            if cached and isinstance(cached.get('portfolio_data'), dict):
                portfolio_data = dict(cached['portfolio_data'])
                cache_age_seconds = (datetime.now(timezone.utc) - cached['last_fetched_at']).total_seconds()
                
                portfolio_data["cached"] = True
                portfolio_data["cache_age_minutes"] = round(cache_age_seconds / 60, 1)
                portfolio_data["last_fetched_at"] = cached['last_fetched_at'].isoformat()
                
                return portfolio_data
        except Exception as e:
            print(f"Cache retrieval error: {e}")
    
    # Fetch fresh data
    print(f"Fetching fresh portfolio data for user {clerk_user_id}")
    portfolio_data = await fetch_portfolio_from_snaptrade(clerk_user_id)
    
    if not isinstance(portfolio_data, dict):
        return {
            "connected": False,
            "error": "Invalid response",
            "accounts": [],
            "positions": [],
            "trades": [],
            "totalNav": 0.0,
            "cached": False,
            "cache_age_minutes": 0,
            "last_fetched_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Update cache
    connection_status = "connected" if portfolio_data.get("connected") and not portfolio_data.get("error") else "error" if portfolio_data.get("error") else "disconnected"
    
    try:
        storage_data = {k: v for k, v in portfolio_data.items() if k not in ['cached', 'cache_age_minutes', 'last_fetched_at']}
        await db.update_portfolio_cache(clerk_user_id, storage_data, connection_status)
    except Exception as e:
        print(f"Failed to update portfolio cache: {e}")
    
    portfolio_data["cached"] = False
    portfolio_data["cache_age_minutes"] = 0
    portfolio_data["last_fetched_at"] = datetime.now(timezone.utc).isoformat()
    
    return portfolio_data