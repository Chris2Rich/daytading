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

def compute_total_nav(
    positions: list,
    accounts: list,
    base_currency: str = "USD"
) -> float:
    """
    Compute total NAV including securities, cash, and FX.

    - Securities: price × quantity
    - Cash: from account balances
    - FX: converted when rate available, otherwise skipped
    """

    total = 0.0

    # 1️⃣ Securities from positions
    for pos in positions or []:
        price = safe_float(pos.get("price"))
        quantity = safe_float(pos.get("fractional_units"))

        if price <= 0 or quantity == 0:
            continue

        market_value = price * abs(quantity)
        total += market_value

    # 2️⃣ Cash + FX from account balances
    for acc in accounts or []:
        balance = acc.get("balance") or {}

        # Total balance is safest if provided
        total_balance = safe_float(balance.get("total"))
        if total_balance > 0:
            total += total_balance
            continue

        # Otherwise fall back to cash
        cash = safe_float(balance.get("cash"))
        if cash > 0:
            total += cash

        # FX balances (optional, not always present)
        fx_balances = balance.get("fx_balances") or []
        for fx in fx_balances:
            amount = safe_float(fx.get("amount"))
            rate = safe_float(fx.get("fx_rate_to_base"))

            if amount > 0 and rate > 0:
                total += amount * rate

    return round(total, 2)

def compute_nav(accounts, positions):
    # Preferred: broker-reported totals
    total = sum(
        safe_float(acc.get("balance", {}).get("total"))
        for acc in accounts or []
    )

    if total > 0:
        return round(total, 2)

    # Fallback
    return compute_total_nav(positions, accounts)

def format_orders(orders: list) -> list:
    """Format order data into standardized trade format"""
    if not isinstance(orders, list):
        return []
    
    trades = []
    for order in orders:
        # Skip orders that aren't filled
        status = (order.get("status") or "").lower()
        if status not in ["filled", "executed", "partial_fill", "partially_filled"]:
            continue
        
        # Extract symbol information
        symbol_obj = order.get("symbol", {})
        if isinstance(symbol_obj, dict):
            symbol = symbol_obj.get("symbol") or symbol_obj.get("ticker")
        else:
            symbol = str(symbol_obj) if symbol_obj else None
        
        # Determine order side (buy/sell)
        action = order.get("action") or order.get("order_type") or ""
        side = action.upper() if action else "UNKNOWN"
        
        # Extract filled quantity and price
        filled_quantity = abs(
            safe_float(order.get("filled_units")) or 
            safe_float(order.get("filled_quantity")) or 
            safe_float(order.get("units")) or 
            safe_float(order.get("quantity"))
        )
        
        # Use average fill price if available, otherwise use limit price or price
        price = (
            safe_float(order.get("average_fill_price")) or
            safe_float(order.get("limit_price")) or
            safe_float(order.get("price"))
        )
        
        # Calculate total amount
        amount = filled_quantity * price if price > 0 else 0.0
        
        # Get timestamp - prefer execution time over order time
        timestamp = (
            order.get("execution_date") or 
            order.get("updated_at") or 
            order.get("time_placed") or
            order.get("order_date")
        )
        
        trades.append({
            "id": order.get("id"),
            "date": timestamp,
            "timestamp": timestamp,
            "side": side,
            "type": side,
            "symbol": symbol,
            "ticker": symbol,
            "quantity": filled_quantity,
            "qty": filled_quantity,
            "price": price,
            "amount": amount,
            "fee": safe_float(order.get("commission", 0)),
            "status": status,
            "order_type": order.get("order_type"),
            "time_in_force": order.get("time_in_force"),
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

        # Fetch authorizations
        authorizations_resp = snaptrade.connections.list_brokerage_authorizations(
            user_id=snaptrade_user_id,
            user_secret=snaptrade_user_secret
        )
        
        authorizations = authorizations_resp.body if hasattr(authorizations_resp, 'body') else authorizations_resp
        
        # Fetch accounts
        accounts_resp = snaptrade.account_information.list_user_accounts(
            user_id=snaptrade_user_id,
            user_secret=snaptrade_user_secret
        )
        
        accounts = accounts_resp.body if hasattr(accounts_resp, 'body') else accounts_resp
        
        if not accounts:
            return {
                "connected": True,
                "accounts": [],
                "positions": [],
                "trades": [],
                "totalNav": 0.0
            }
        
        # Map accounts to authorizations by brokerage name
        account_to_auth = {}
        for account in accounts or []:
            account_id = account.get("id")
            account_institution = account.get("institution_name") or ""
            
            for auth in authorizations or []:
                auth_id = auth.get("id")
                auth_brokerage = auth.get("brokerage", {}).get("name", "")
                
                if account_institution.lower() == auth_brokerage.lower():
                    account_to_auth[account_id] = auth_id
                    break
        
        # Fetch positions and orders for each account
        all_positions = []
        all_orders = []
        total_cash = 0.0
        total_holdings_value = 0.0

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
                    
                    # Calculate holdings market value
                    for position in positions:
                        price = safe_float(position.get("price", 0))
                        quantity = safe_float(position.get("fractional_units") or position.get("units", 0))
                        market_value = price * abs(quantity)
                        total_holdings_value += market_value
                
                # Get account balance for cash
                balance_resp = snaptrade.account_information.get_user_account_balance(
                    user_id=snaptrade_user_id,
                    user_secret=snaptrade_user_secret,
                    account_id=account_id
                )
                balance_data = balance_resp.body if hasattr(balance_resp, 'body') else balance_resp
                
                # Sum cash from all currencies
                if isinstance(balance_data, list):
                    for balance_item in balance_data:
                        cash = safe_float(balance_item.get("cash", 0))
                        total_cash += cash
                
                # Get account orders
                try:
                    orders_resp = snaptrade.account_information.get_user_account_orders(
                        user_id=snaptrade_user_id,
                        user_secret=snaptrade_user_secret,
                        account_id=account_id,
                        state="all"  # Get all orders regardless of status
                    )
                    orders = orders_resp.body if hasattr(orders_resp, 'body') else orders_resp
                    if orders:
                        all_orders.extend(orders)
                except Exception as e:
                    print(f"Error fetching orders for account {account_id}: {e}")
                    
            except Exception as e:
                print(f"Error fetching data for account {account_id}: {e}")
                continue
        
        # Calculate total NAV = Holdings + Cash
        total_nav = total_holdings_value + total_cash
        
        # Process orders into trade format
        trades = format_orders(all_orders)
        
        # Format accounts
        formatted_accounts = [
            {
                "id": acc.get("id"),
                "name": acc.get("name"),
                "accountNumber": acc.get("number"),
                "brokerage": acc.get("institution_name") or acc.get("brokerage_name"),
                "type": acc.get("type"),
                "authorizationId": account_to_auth.get(acc.get("id")),
            }
            for acc in accounts
        ]
        
        return {
            "connected": True,
            "accounts": formatted_accounts,
            "positions": all_positions,
            "trades": trades,
            "totalNav": round(total_nav, 2)
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

async def remove_broker_authorization(
    clerk_user_id: str,
    authorization_id: str,
    db
) -> Dict[str, Any]:
    """
    Remove a broker authorization from SnapTrade and invalidate cache.
    
    Args:
        clerk_user_id: Clerk user ID
        authorization_id: SnapTrade authorization ID to remove
        db: Database instance for cache invalidation
    
    Returns:
        Dict with success status and message
    """
    if not SNAPTRADE_CLIENT_ID or not SNAPTRADE_SECRET:
        return {
            "success": False,
            "error": "SnapTrade not configured"
        }
    
    if not SnapTrade:
        return {
            "success": False,
            "error": "SnapTrade SDK not installed"
        }
    
    try:
        # Get SnapTrade credentials from Clerk
        user_response = clerk.users.get(user_id=clerk_user_id)
        private_metadata = user_response.private_metadata or {}
        
        snaptrade_user_id = private_metadata.get("snaptradeUserId", clerk_user_id)
        snaptrade_user_secret = private_metadata.get("snaptradeUserSecret")
        
        if not snaptrade_user_secret:
            return {
                "success": False,
                "error": "SnapTrade not connected for this user"
            }
        
        # Initialize SnapTrade SDK client
        snaptrade = SnapTrade(
            client_id=SNAPTRADE_CLIENT_ID,
            consumer_key=SNAPTRADE_SECRET
        )
        
        # Remove the authorization from SnapTrade
        try:
            snaptrade.connections.remove_brokerage_authorization(
                authorization_id=authorization_id,
                user_id=snaptrade_user_id,
                user_secret=snaptrade_user_secret
            )
        except Exception as snaptrade_error:
            print(f"SnapTrade authorization removal error: {snaptrade_error}")
            # Continue to cache invalidation even if SnapTrade call fails
        
        # Invalidate portfolio cache
        await db.invalidate_portfolio_cache(clerk_user_id)
        
        # Update Clerk metadata to remove broker from list
        public_metadata = user_response.public_metadata or {}
        existing_brokers = public_metadata.get("brokers", [])
        
        if isinstance(existing_brokers, list) and authorization_id in existing_brokers:
            updated_brokers = [b for b in existing_brokers if b != authorization_id]
            
            clerk.users.update_metadata(
                user_id=clerk_user_id,
                public_metadata={
                    **public_metadata,
                    "brokers": updated_brokers
                }
            )
        
        return {
            "success": True,
            "message": f"Broker authorization {authorization_id} removed successfully"
        }
    
    except Exception as e:
        print(f"Error removing broker authorization: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }