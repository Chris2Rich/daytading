import os
import httpx
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from clerk_backend_api import Clerk

# SnapTrade configuration
SNAPTRADE_CLIENT_ID = os.environ.get("SNAPTRADE_CLIENT_ID")
SNAPTRADE_SECRET = os.environ.get("SNAPTRADE_SECRET")
SNAPTRADE_BASE_URL = "https://api.snaptrade.com/api/v1"

if not SNAPTRADE_CLIENT_ID or not SNAPTRADE_SECRET:
    print("Warning: SnapTrade credentials not configured")

# Initialize Clerk SDK
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)


def compute_total_nav_from_positions(positions: list) -> float:
    """Calculate total NAV from positions"""
    if not isinstance(positions, list):
        return 0.0
    
    total = 0.0
    for position in positions:
        price = float(position.get("price", 0))
        quantity = float(position.get("fractional_units", 0))
        market_value = price * abs(quantity)
        total += market_value
    
    return total


def format_trades(activities: list) -> list:
    """Filter and format trade activities"""
    if not isinstance(activities, list):
        return []
    
    trades = []
    for activity in activities:
        activity_type = (activity.get("type") or "").lower()
        if "buy" in activity_type or "sell" in activity_type or "trade" in activity_type:
            trades.append({
                "date": activity.get("trade_date") or activity.get("settlement_date") or activity.get("date"),
                "timestamp": activity.get("trade_date") or activity.get("settlement_date"),
                "side": activity.get("type"),
                "type": activity.get("type"),
                "symbol": (activity.get("symbol") or {}).get("symbol") if isinstance(activity.get("symbol"), dict) else activity.get("symbol"),
                "ticker": (activity.get("symbol") or {}).get("symbol") if isinstance(activity.get("symbol"), dict) else activity.get("symbol"),
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
    Fetch portfolio data from SnapTrade API.
    
    Returns dict with:
        - accounts: list of account objects
        - positions: list of position objects
        - trades: list of formatted trade objects
        - total_nav: float
        - connected: bool
        - error: str (if error occurred)
    """
    if not SNAPTRADE_CLIENT_ID or not SNAPTRADE_SECRET:
        return {
            "connected": False,
            "error": "SnapTrade not configured",
            "accounts": [],
            "positions": [],
            "trades": [],
            "total_nav": 0.0
        }
    
    try:
        # Get SnapTrade credentials from Clerk
        user = clerk.users.get(user_id=clerk_user_id)
        snaptrade_user_id = user.private_metadata.get("snaptradeUserId") or clerk_user_id
        snaptrade_user_secret = user.private_metadata.get("snaptradeUserSecret")
        
        if not snaptrade_user_secret:
            return {
                "connected": False,
                "error": "SnapTrade not connected",
                "accounts": [],
                "positions": [],
                "trades": [],
                "total_nav": 0.0
            }
        
        # Create HTTP client with SnapTrade auth
        headers = {
            "Content-Type": "application/json",
        }
        
        # SnapTrade uses HTTP Basic Auth with clientId:consumerKey
        auth = (SNAPTRADE_CLIENT_ID, SNAPTRADE_SECRET)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch accounts
            accounts_resp = await client.get(
                f"{SNAPTRADE_BASE_URL}/accounts",
                params={
                    "userId": snaptrade_user_id,
                    "userSecret": snaptrade_user_secret
                },
                headers=headers,
                auth=auth
            )
            accounts_resp.raise_for_status()
            accounts = accounts_resp.json()
            
            if not accounts:
                return {
                    "connected": True,
                    "accounts": [],
                    "positions": [],
                    "trades": [],
                    "total_nav": 0.0
                }
            
            # Fetch positions and activities for each account
            all_positions = []
            all_activities = []
            
            for account in accounts:
                account_id = account.get("id")
                
                try:
                    # Get positions
                    positions_resp = await client.get(
                        f"{SNAPTRADE_BASE_URL}/accounts/{account_id}/positions",
                        params={
                            "userId": snaptrade_user_id,
                            "userSecret": snaptrade_user_secret
                        },
                        headers=headers,
                        auth=auth
                    )
                    positions_resp.raise_for_status()
                    positions = positions_resp.json()
                    all_positions.extend(positions)
                    
                    # Get activities
                    activities_resp = await client.get(
                        f"{SNAPTRADE_BASE_URL}/activities",
                        params={
                            "userId": snaptrade_user_id,
                            "userSecret": snaptrade_user_secret,
                            "accountId": account_id
                        },
                        headers=headers,
                        auth=auth
                    )
                    activities_resp.raise_for_status()
                    activities_data = activities_resp.json()
                    activities = activities_data if isinstance(activities_data, list) else activities_data.get("data", [])
                    all_activities.extend(activities)
                    
                except httpx.HTTPError as e:
                    print(f"Error fetching data for account {account_id}: {e}")
                    continue
            
            # Process data
            total_nav = compute_total_nav_from_positions(all_positions)
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
                "total_nav": total_nav
            }
    
    except httpx.HTTPError as e:
        print(f"SnapTrade API error: {e}")
        return {
            "connected": True,
            "error": f"SnapTrade API error: {str(e)}",
            "accounts": [],
            "positions": [],
            "trades": [],
            "total_nav": 0.0
        }
    except Exception as e:
        print(f"Error fetching portfolio from SnapTrade: {e}")
        return {
            "connected": False,
            "error": str(e),
            "accounts": [],
            "positions": [],
            "trades": [],
            "total_nav": 0.0
        }


async def get_portfolio_data(
    clerk_user_id: str,
    db,  # Database instance
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get portfolio data with intelligent caching.
    
    Args:
        clerk_user_id: User's Clerk ID
        db: Database instance
        force_refresh: If True, bypass cache and fetch fresh data
    
    Returns:
        Portfolio data dict with cache metadata
    """
    cache_max_age_minutes = 20
    
    # Check if force refresh requested
    if not force_refresh:
        # Try to get from cache
        cached = await db.get_portfolio_cache(clerk_user_id, max_age_minutes=cache_max_age_minutes)
        
        if cached:
            # Cache hit - return cached data
            portfolio_data = cached['portfolio_data']
            cache_age_seconds = (datetime.now(timezone.utc) - cached['last_fetched_at']).total_seconds()
            
            # FIXED: Add these fields to cached dict instead of unpacking
            portfolio_data["cached"] = True
            portfolio_data["cache_age_minutes"] = round(cache_age_seconds / 60, 1)
            portfolio_data["last_fetched_at"] = cached['last_fetched_at'].isoformat()
            
            return portfolio_data
    
    # Cache miss or force refresh - fetch from SnapTrade
    print(f"Fetching fresh portfolio data for user {clerk_user_id}")
    portfolio_data = await fetch_portfolio_from_snaptrade(clerk_user_id)
    
    # FIXED: Ensure portfolio_data is a dict
    if not isinstance(portfolio_data, dict):
        print(f"ERROR: fetch_portfolio_from_snaptrade returned non-dict: {type(portfolio_data)}")
        return {
            "connected": False,
            "error": "Invalid response from portfolio fetch",
            "accounts": [],
            "positions": [],
            "trades": [],
            "total_nav": 0.0,
            "cached": False,
            "cache_age_minutes": 0,
            "last_fetched_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Update cache if successful
    if portfolio_data.get("connected") and not portfolio_data.get("error"):
        connection_status = "connected"
    elif portfolio_data.get("error"):
        connection_status = "error"
    else:
        connection_status = "disconnected"
    
    try:
        await db.update_portfolio_cache(
            clerk_user_id,
            portfolio_data,
            connection_status
        )
    except Exception as e:
        print(f"Failed to update portfolio cache: {e}")
    
    # FIXED: Add these fields directly instead of unpacking
    portfolio_data["cached"] = False
    portfolio_data["cache_age_minutes"] = 0
    portfolio_data["last_fetched_at"] = datetime.now(timezone.utc).isoformat()
    
    return portfolio_data