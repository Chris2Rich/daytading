import asyncio
import ssl
import pathlib
import sys
import os
import traceback
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4, UUID
from functools import wraps
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import jwt
from jwt import PyJWKClient
from aiohttp import web
from clerk_backend_api import Clerk

# Setup path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from llm.lmwrapper import LM_WRAPPER, LM_WRAPPER_WITH_TOOLS, routers
from db.database import Database
from retrieval.yahoofinance import (
    get_trading, 
    get_option, 
    get_option_chain, 
    get_fundamentals, 
    get_price_targets,
    get_key_metrics_timeseries
)
from retrieval.snaptrade import get_portfolio_data, remove_broker_authorization
from strategy.main import run_backtest
from strategy.adapters import YFinanceAdapter
from strategy.models import RandomForestModel, ModelOutputType
from strategy.features import rsi_feature, moving_average_feature
from strategy.position_sizers import FixedFractionalSizer
from retrieval.rss_feeds import get_news_fetcher

# Clerk configuration
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
CLERK_JWKS_URL = "https://artistic-hedgehog-31.clerk.accounts.dev/.well-known/jwks.json"
if not CLERK_SECRET_KEY:
    raise ValueError("CLERK_SECRET_KEY environment variable is required")

# Initialize Clerk SDK
clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)

# Initialize JWKS client for token verification
jwks_client = PyJWKClient(CLERK_JWKS_URL)

# Initialize LLM and Database
lm = LM_WRAPPER_WITH_TOOLS(routers["GROQ"]["models"]["llama4scout"])
titlelm = LM_WRAPPER(routers["GROQ"]["models"]["llama8b"])
db = Database()

# Initialize news fetcher with database for caching
from retrieval.rss_feeds import get_news_fetcher
news_fetcher = None


def serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to JSON-serializable strings.
    Handles UUID, datetime, Decimal, and other types.
    """
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


async def verify_clerk_token(token: str) -> Optional[str]:
    """
    Verify Clerk JWT token and return clerk_user_id if valid.
    Returns None if invalid.
    """
    try:
        # Get signing key from JWKS
        signing_key = await asyncio.to_thread(
            jwks_client.get_signing_key_from_jwt,
            token
        )
        
        # Verify and decode the token
        decoded = await asyncio.to_thread(
            jwt.decode,
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": True}
        )
        
        # Extract user ID from 'sub' claim
        clerk_user_id = decoded.get("sub")
        return clerk_user_id
        
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None
    except Exception as e:
        print(f"Token verification failed: {e}")
        traceback.print_exc()
        return None


def require_auth(handler):
    """Decorator to require Clerk authentication"""
    @wraps(handler)
    async def wrapper(request):
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return web.json_response(
                {"error": "Missing or invalid Authorization header"},
                status=401
            )
        
        token = auth_header[7:]
        clerk_user_id = await verify_clerk_token(token)
        
        if not clerk_user_id:
            return web.json_response(
                {"error": "Invalid token"},
                status=401
            )
        
        # Ensure user exists in database
        await db.ensure_user(clerk_user_id)
        
        request["clerk_user_id"] = clerk_user_id
        return await handler(request)
    
    return wrapper


@require_auth
async def handle_generate(request):
    """
    POST /lm/generate
    Stream LLM response with SSE
    
    Request body: {
        "message": str,
        "tools": List[str],
        "chatid": Optional[str]
    }
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        tools = data.get("tools", [])
        chat_id = data.get("chatid")

        if not message:
            return web.json_response(
                {"error": "Message is required"},
                status=400
            )

        is_new_chat = False
        
        # Create new chat if no chat_id provided
        if not chat_id:
            chat_id = await db.create_chat(clerk_user_id)
            is_new_chat = True
        else:
            # Verify chat ownership
            if not await db.verify_chat_ownership(chat_id, clerk_user_id):
                return web.json_response(
                    {"error": "Chat not found or access denied"},
                    status=403
                )
        
        # Store user message
        user_message_id = await db.add_message(
            chat_id=chat_id,
            clerk_user_id=clerk_user_id,
            role="user",
            content=message,
            tools=tools
        )
        
        # Get full chat history
        chat_history = await db.get_chat_history(chat_id, clerk_user_id)
        
        # Prepare SSE response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
        await response.prepare(request)
        
        # Send initial metadata
        await response.write(
            f"data: {json.dumps({'type': 'metadata', 'chatid': chat_id, 'message_id': user_message_id})}\n\n".encode()
        )
        
        # Stream LLM response with search support
        assistant_response = ""
        pending_thoughts = []
        pending_tool_calls = []
        try:
            async for token_data in lm.stream_agentic(
                user_message=message,
                chat_history=chat_history[:-1],
                requested_tools=tools,
                max_iterations=5,
                clerk_user_id=clerk_user_id,
                db=db,
            ):
                if token_data["type"] == "thinking":
                    await response.write(
                        f"data: {json.dumps({'type': 'thinking', 'content': token_data.get('content', '')})}\n\n".encode()
                    )
                    pending_thoughts.append(token_data.get("content", ""))
                    
                elif token_data["type"] == "thinking_chunk":
                    await response.write(
                        f"data: {json.dumps({'type': 'thinking_chunk', 'content': token_data.get('content', '')})}\n\n".encode()
                    )
                    
                elif token_data["type"] == "tool_call":
                    tool_event = {
                        "type": "tool_call",
                        "tool": token_data.get("tool", ""),
                        "args": token_data.get("args", {}),
                    }
                    await response.write(
                        f"data: {json.dumps(tool_event)}\n\n".encode()
                    )
                    await asyncio.sleep(0.1)
                    
                elif token_data["type"] == "tool_result":
                    pending_tool_calls.append({
                        "tool": token_data.get("tool", ""),
                        "result": token_data.get("result", ""),
                    })
                    tool_result_event = {
                        "type": "tool_result",
                        "tool": token_data.get("tool", ""),
                        "result": token_data.get("result", "")[:500],
                    }
                    await response.write(
                        f"data: {json.dumps(tool_result_event)}\n\n".encode()
                    )
                    await asyncio.sleep(0.1)
                    
                elif token_data["type"] == "token":
                    token = token_data["content"]
                    assistant_response += token
                    await response.write(
                        f"data: {json.dumps({'type': 'token', 'content': token})}\n\n".encode()
                    )
                
                elif token_data["type"] == "thoughts_complete":
                    await response.write(
                        f"data: {json.dumps({'type': 'thoughts_complete', 'thoughts': token_data.get('thoughts', [])})}\n\n".encode()
                    )
                
                elif token_data["type"] == "tool_log":
                    await response.write(
                        f"data: {json.dumps({'type': 'tool_log', 'calls': token_data.get('calls', [])})}\n\n".encode()
                    )
                
                elif token_data["type"] == "error":
                    await response.write(
                        f"data: {json.dumps({'type': 'error', 'error': token_data.get('error', 'Unknown error')})}\n\n".encode()
                    )
            
            # Store assistant message
            assistant_message_id = await db.add_message(
                chat_id=chat_id,
                clerk_user_id=clerk_user_id,
                role="assistant",
                content=assistant_response
            )

            # Generate title for new chats
            if is_new_chat:
                asyncio.create_task(generate_and_update_title(chat_id, message))
            
            # Send completion event
            await response.write(
                f"data: {json.dumps({'type': 'done', 'message_id': assistant_message_id})}\n\n".encode()
            )
            
        except asyncio.CancelledError:
            # Client disconnected
            print(f"Client disconnected during streaming for chat {chat_id}")
        except Exception as e:
            error_id = str(chat_id)
            print(f"Error {error_id} during streaming: {e}")
            traceback.print_exc()
            
            await response.write(
                f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n".encode()
            )
        
        await response.write_eof()
        return response
        
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        print(f"Error in handle_generate: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


async def generate_and_update_title(chat_id: str, first_message: str):
    """Background task to generate and update chat title"""
    try:
        title = await titlelm.generate_title(first_message)
        await db.update_chat_title(chat_id, title)
    except Exception as e:
        print(f"Error generating title: {e}")


@require_auth
async def handle_get_history(request):
    """
    GET /lm/gethistory?chatid=<chat_id>
    Return ordered chat history
    """
    clerk_user_id = request["clerk_user_id"]
    chat_id = request.query.get("chatid")
    
    if not chat_id:
        return web.json_response(
            {"error": "chatid query parameter is required"},
            status=400
        )
    
    try:
        history = await db.get_chat_history(chat_id, clerk_user_id)
        
        # Serialize all non-JSON-serializable objects
        history = [serialize_for_json(msg) for msg in history]
        
        return web.json_response({
            "chatid": chat_id,
            "messages": history
        })
        
    except ValueError as e:
        return web.json_response(
            {"error": str(e)},
            status=404
        )
    except PermissionError as e:
        return web.json_response(
            {"error": str(e)},
            status=403
        )
    except Exception as e:
        print(f"Error in handle_get_history: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_chats(request):
    """
    GET /lm/chats
    List all user's chats
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        chats = await db.get_user_chats(clerk_user_id)
        
        # Serialize all non-JSON-serializable objects
        chats = [serialize_for_json(chat) for chat in chats]
        
        return web.json_response({
            "chats": chats
        })
        
    except Exception as e:
        print(f"Error in handle_get_chats: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_delete_chat(request):
    """
    DELETE /lm/deletechat?chatid=<chat_id>
    Delete a chat and all its messages
    """
    clerk_user_id = request["clerk_user_id"]
    chat_id = request.query.get("chatid")
    
    if not chat_id:
        return web.json_response(
            {"error": "chatid query parameter is required"},
            status=400
        )
    
    try:
        # Verify chat ownership
        if not await db.verify_chat_ownership(chat_id, clerk_user_id):
            return web.json_response(
                {"error": "Chat not found or access denied"},
                status=403
            )
        
        # Delete the chat (messages will cascade delete)
        await db.delete_chat(chat_id)
        
        return web.json_response({
            "success": True,
            "chatid": chat_id
        })
        
    except Exception as e:
        print(f"Error in handle_delete_chat: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_ticker_data(request):
    """
    GET /data/tickerdata
    
    Query parameters:
    - action: trading|option|option_chain|fundamentals|price_targets
    - ticker: Stock symbol (required)
    
    For trading:
    - days: Number of days (default: 30)
    - interval: 1m|5m|15m|30m|1h|1d|1mo (default: 1d)
    
    For option:
    - expiry: YYMMDD format (required)
    - direction: C|P (required)
    - strike: Strike price (required)
    
    For option_chain:
    - expiry: YYMMDD format (required)
    
    For fundamentals:
    - statement_type: income_statement|balance_sheet (required)
    - period: annual|quarterly (default: annual)
    """
    try:
        action = request.query.get("action")
        ticker = request.query.get("ticker")
        
        if not action:
            return web.json_response(
                {"error": "action parameter is required"},
                status=400
            )
        
        if not ticker:
            return web.json_response(
                {"error": "ticker parameter is required"},
                status=400
            )
        
        # Execute action in thread to avoid blocking
        if action == "trading":
            days = int(request.query.get("days", "30"))
            interval = request.query.get("interval", "1d")
            
            data = await asyncio.to_thread(
                get_trading,
                ticker,
                days,
                interval
            )
            
            # Convert DataFrame to JSON-serializable format
            result = {
                "ticker": ticker,
                "interval": interval,
                "data": [
                    {
                        "date": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        "open": float(row["Open"].iloc[0]),
                        "high": float(row["High"].iloc[0]),
                        "low": float(row["Low"].iloc[0]),
                        "close": float(row["Close"].iloc[0]),
                        "volume": int(row["Volume"].iloc[0]) if "Volume" in row else 0,
                    }
                    for idx, row in data.iterrows()
                ]
            }
            
            return web.json_response(result)

        
        
        elif action == "option":
            expiry = request.query.get("expiry")
            direction = request.query.get("direction")
            strike = request.query.get("strike")
            
            if not all([expiry, direction, strike]):
                return web.json_response(
                    {"error": "expiry, direction, and strike are required for option"},
                    status=400
                )
            
            data = await asyncio.to_thread(
                get_option,
                ticker,
                expiry,
                direction,
                float(strike)
            )

            data = data.reset_index()
            
            result = {
                "ticker": ticker,
                "expiry": expiry,
                "direction": direction,
                "strike": float(strike),
                "data": [
                    {
                        "date": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        "open": float(row["Open"].iloc[0]),
                        "high": float(row["High"].iloc[0]),
                        "low": float(row["Low"].iloc[0]),
                        "close": float(row["Close"].iloc[0]),
                        "volume": int(row["Volume"].iloc[0]) if "Volume" in row else 0,
                    }
                    for idx, row in data.iterrows()
                ]
            }
            
            return web.json_response(result)
        
        elif action == "option_chain":
            expiry = request.query.get("expiry")
            
            if not expiry:
                return web.json_response(
                    {"error": "expiry is required for option_chain"},
                    status=400
                )
            
            data = await asyncio.to_thread(
                get_option_chain,
                ticker,
                expiry
            )
            
            return web.json_response({
                "ticker": ticker,
                "expiry": expiry,
                "calls": data["calls"],
                "puts": data["puts"]
            })
        
        elif action == "fundamentals":
            statement_type = request.query.get("statement_type")
            period = request.query.get("period", "annual")
            
            if not statement_type:
                return web.json_response(
                    {"error": "statement_type is required for fundamentals"},
                    status=400
                )
            
            data = await asyncio.to_thread(
                get_fundamentals,
                ticker,
                statement_type,
                period
            )
            
            # Convert timestamps to ISO format
            formatted_data = {}
            for key, value in data.items():
                if hasattr(key, 'isoformat'):
                    formatted_data[key.isoformat()] = value
                else:
                    formatted_data[str(key)] = value
            
            return web.json_response({
                "ticker": ticker,
                "statement_type": statement_type,
                "period": period,
                "data": formatted_data
            })
        
        elif action == "price_targets":
            data = await asyncio.to_thread(
                get_price_targets,
                ticker
            )
            
            # Convert DataFrame to dict if present
            if 'recommendation_history' in data and hasattr(data['recommendation_history'], 'to_dict'):
                data['recommendation_history'] = data['recommendation_history'].to_dict(orient='records')
            
            return web.json_response(data)

        elif action == "key_metrics":
            period = request.query.get("period", "quarterly")
        
            data = await asyncio.to_thread(
                get_key_metrics_timeseries,
                ticker,
                period
            )
            
            return web.json_response({
                "ticker": ticker,
                "period": period,
                "metrics": data
            })
                
        else:
            return web.json_response(
                {"error": f"Unknown action: {action}"},
                status=400
            )
    
    except Exception as e:
        print(f"Error in handle_ticker_data: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": str(e)},
            status=500
        )

@require_auth
async def handle_portfolio_data(request):
    """
    GET /data/portfolio
    
    Query parameters:
    - refresh: true|false (default: false) - Force refresh from SnapTrade
    
    Returns portfolio data with caching metadata
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        refresh_param = request.query.get("refresh")
        force_refresh = str(refresh_param).lower() == "true" if refresh_param is not None else False

        portfolio_data = await get_portfolio_data(
            clerk_user_id=clerk_user_id,
            db=db,
            force_refresh=force_refresh
        )
        
        return web.json_response(portfolio_data)
    
    except Exception as e:
        print(f"Error in handle_portfolio_data: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": str(e)},
            status=500
        )

@require_auth
async def handle_remove_broker(request):
    """
    POST /data/broker/remove
    
    Request body: {
        "authorizationId": str
    }
    
    Removes broker authorization from SnapTrade and invalidates cache.
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        data = await request.json()
        authorization_id = data.get("authorizationId")
        
        if not authorization_id:
            return web.json_response(
                {"error": "authorizationId is required"},
                status=400
            )
        
        result = await remove_broker_authorization(
            clerk_user_id=clerk_user_id,
            authorization_id=authorization_id,
            db=db
        )
        
        if result.get("success"):
            return web.json_response(result)
        else:
            return web.json_response(
                {"error": result.get("error", "Failed to remove broker")},
                status=500
            )
    
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        print(f"Error in handle_remove_broker: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": str(e)},
            status=500
        )


# ============================================================================
# Strategy Endpoints
# ============================================================================

@require_auth
async def handle_list_strategies(request):
    """
    GET /strategies
    List all strategies for the authenticated user
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        strategies = await db.get_user_strategies(clerk_user_id)
        
        # Serialize all non-JSON-serializable objects
        strategies = [serialize_for_json(s) for s in strategies]
        
        return web.json_response({"strategies": strategies})
    
    except Exception as e:
        print(f"Error in handle_list_strategies: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_create_strategy(request):
    """
    POST /strategies
    Create a new strategy
    
    Request body: {
        "name": str,
        "description": str,
        "config": StrategyConfig,
        "tags": [str]
    }
    """
    clerk_user_id = request["clerk_user_id"]
    
    try:
        data = await request.json()
        
        name = data.get("name")
        description = data.get("description", "")
        config = data.get("config")
        tags = data.get("tags", [])
        
        if not name:
            return web.json_response(
                {"error": "name is required"},
                status=400
            )
        
        if not config:
            return web.json_response(
                {"error": "config is required"},
                status=400
            )
        
        strategy_id = await db.create_strategy(
            clerk_user_id=clerk_user_id,
            name=name,
            description=description,
            config=config,
            tags=tags
        )
        
        return web.json_response({
            "success": True,
            "strategy_id": strategy_id
        }, status=201)
    
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        print(f"Error in handle_create_strategy: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_strategy(request):
    """
    GET /strategies/{strategy_id}
    Get a specific strategy
    """
    clerk_user_id = request["clerk_user_id"]
    strategy_id = request.match_info.get('strategy_id')
    
    if not strategy_id:
        return web.json_response(
            {"error": "strategy_id is required"},
            status=400
        )
    
    try:
        strategy = await db.get_strategy(strategy_id, clerk_user_id)
        
        if not strategy:
            return web.json_response(
                {"error": "Strategy not found"},
                status=404
            )
        
        # Include latest backtest results if available
        latest_backtest = await db.get_latest_backtest(strategy_id, clerk_user_id)
        if latest_backtest:
            strategy['backtest_results'] = serialize_for_json(latest_backtest)
        
        # Serialize all non-JSON-serializable objects
        strategy = serialize_for_json(strategy)
        
        return web.json_response(strategy)
    
    except Exception as e:
        print(f"Error in handle_get_strategy: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_update_strategy(request):
    """
    PUT /strategies/{strategy_id}
    Update a strategy
    """
    clerk_user_id = request["clerk_user_id"]
    strategy_id = request.match_info.get('strategy_id')
    
    if not strategy_id:
        return web.json_response(
            {"error": "strategy_id is required"},
            status=400
        )
    
    try:
        data = await request.json()
        
        # Verify ownership
        if not await db.verify_strategy_ownership(strategy_id, clerk_user_id):
            return web.json_response(
                {"error": "Strategy not found or access denied"},
                status=403
            )
        
        # Update fields that are provided
        success = await db.update_strategy(
            strategy_id=strategy_id,
            clerk_user_id=clerk_user_id,
            name=data.get('name'),
            description=data.get('description'),
            config=data.get('config'),
            status=data.get('status'),
            tags=data.get('tags')
        )
        
        if success:
            return web.json_response({"success": True})
        else:
            return web.json_response(
                {"error": "Failed to update strategy"},
                status=500
            )
    
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        print(f"Error in handle_update_strategy: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_delete_strategy(request):
    """
    DELETE /strategies/{strategy_id}
    Delete a strategy
    """
    clerk_user_id = request["clerk_user_id"]
    strategy_id = request.match_info.get('strategy_id')
    
    if not strategy_id:
        return web.json_response(
            {"error": "strategy_id is required"},
            status=400
        )
    
    try:
        success = await db.delete_strategy(strategy_id, clerk_user_id)
        
        if success:
            return web.json_response({"success": True})
        else:
            return web.json_response(
                {"error": "Strategy not found or access denied"},
                status=404
            )
    
    except Exception as e:
        print(f"Error in handle_delete_strategy: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_run_backtest(request):
    """
    POST /strategies/{strategy_id}/backtest
    Run a backtest for a strategy (async - returns job ID)
    
    Request body: {
        "backtest_id": str (optional - for retry)
    }
    """
    clerk_user_id = request["clerk_user_id"]
    strategy_id = request.match_info.get('strategy_id')
    
    if not strategy_id:
        return web.json_response(
            {"error": "strategy_id is required"},
            status=400
        )
    
    try:
        # Verify strategy exists and user owns it
        strategy = await db.get_strategy(strategy_id, clerk_user_id)
        if not strategy:
            return web.json_response(
                {"error": "Strategy not found or access denied"},
                status=403
            )
        
        data = await request.json() if request.can_read_body else {}
        backtest_id = data.get('backtest_id')
        
        # Create new backtest record if not retrying
        if not backtest_id:
            backtest_id = await db.create_backtest(strategy_id, clerk_user_id)
        else:
            # Reset existing backtest
            await db.update_backtest_status(backtest_id, clerk_user_id, 'pending')
        
        # Update strategy status
        await db.update_strategy(
            strategy_id=strategy_id,
            clerk_user_id=clerk_user_id,
            status='backtesting'
        )
        
        # Start backtest in background task
        asyncio.create_task(run_backtest_task(backtest_id, strategy_id, clerk_user_id, strategy['config']))
        
        return web.json_response({
            "success": True,
            "backtest_id": backtest_id,
            "status": "pending"
        })
    
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        print(f"Error in handle_run_backtest: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_backtest_status(request):
    """
    GET /backtest/{backtest_id}/status
    Get backtest status
    """
    clerk_user_id = request["clerk_user_id"]
    backtest_id = request.match_info.get('backtest_id')
    
    if not backtest_id:
        return web.json_response(
            {"error": "backtest_id is required"},
            status=400
        )
    
    try:
        backtest = await db.get_backtest(backtest_id, clerk_user_id)
        
        if not backtest:
            return web.json_response(
                {"error": "Backtest not found or access denied"},
                status=404
            )
        
        # Serialize all non-JSON-serializable objects
        backtest = serialize_for_json(backtest)
        
        return web.json_response(backtest)
    
    except Exception as e:
        print(f"Error in handle_get_backtest_status: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_backtest_results(request):
    """
    GET /backtest/{backtest_id}/results
    Get full backtest results including trades
    """
    clerk_user_id = request["clerk_user_id"]
    backtest_id = request.match_info.get('backtest_id')
    
    if not backtest_id:
        return web.json_response(
            {"error": "backtest_id is required"},
            status=400
        )
    
    try:
        backtest = await db.get_backtest(backtest_id, clerk_user_id)
        
        if not backtest:
            return web.json_response(
                {"error": "Backtest not found or access denied"},
                status=404
            )
        
        # Get trades if backtest is completed
        trades = []
        if backtest.get('status') == 'completed':
            trades = await db.get_trades(backtest_id, clerk_user_id, limit=1000)
        
        # Serialize all non-JSON-serializable objects
        backtest = serialize_for_json(backtest)
        backtest['trades'] = serialize_for_json(trades)
        
        return web.json_response(backtest)
    
    except Exception as e:
        print(f"Error in handle_get_backtest_results: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


async def run_backtest_task(backtest_id: str, strategy_id: str, clerk_user_id: str, config: dict):
    """
    Background task to run backtest using NautilusTrader
    """
    try:
        # Update status to running
        await db.update_backtest_status(backtest_id, clerk_user_id, 'running')
        start_time = time.time()
        
        # Validate and prepare config for TradingSystemConfig
        # Convert frontend config format to TradingSystemConfig format
        trading_config = prepare_trading_config(config)
        
        # Run the actual backtest using NautilusTrader
        print(f"Starting backtest {backtest_id} for strategy {strategy_id}")
        engine = run_backtest(trading_config)
        
        # Extract results from engine
        results = extract_backtest_results(engine, trading_config)
        results['duration_seconds'] = int(time.time() - start_time)
        results['starting_capital'] = float(trading_config['starting_capital'])
        results['ending_capital'] = results['portfolio']['endingCapital']
        
        # Store results
        await db.update_backtest_results(backtest_id, clerk_user_id, results)
        
        # Store trades
        if results.get('trades'):
            await db.add_trades(backtest_id, results['trades'])
        
        # Update strategy status
        await db.update_strategy(
            strategy_id=strategy_id,
            clerk_user_id=clerk_user_id,
            status='completed'
        )
        
        print(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        print(f"Backtest {backtest_id} failed: {e}")
        traceback.print_exc()
        
        # Update status to failed
        await db.update_backtest_status(backtest_id, clerk_user_id, 'failed', str(e))
        
        # Update strategy status
        await db.update_strategy(
            strategy_id=strategy_id,
            clerk_user_id=clerk_user_id,
            status='failed'
        )


def prepare_trading_config(config: dict) -> dict:
    """
    Convert frontend config to TradingSystemConfig format
    """
    # Default configuration
    default_config = {
        'seed': 42,
        'id': str(uuid4()),
        'commission_rate': 0.001,
        'max_leverage': 1.0,
        'retrain_frequency': 252,
        'training_window_type': 'expanding',
        'model_save_path': './models'
    }
    
    # Map frontend field names to backend field names
    trading_config = {
        **default_config,
        'traded_symbols': config.get('traded_symbols') or config.get('symbols', ['SPY']),
        'non_traded_timeseries': config.get('non_traded_timeseries', []),
        'backtest_start': parse_datetime(config.get('backtest_start') or config.get('backtestStart', '2020-01-01')),
        'backtest_end': parse_datetime(config.get('backtest_end') or config.get('backtestEnd', '2023-12-31')),
        'starting_capital': Decimal(str(config.get('starting_capital') or config.get('startingCapital', 100000))),
    }
    
    # Setup data adapters for each symbol
    interval = config.get('timeframe', '1d')
    adapters = {}
    for symbol in trading_config['traded_symbols']:
        adapters[symbol] = YFinanceAdapter(interval=interval)
    for symbol in trading_config['non_traded_timeseries']:
        adapters[symbol] = YFinanceAdapter(interval=interval)
    trading_config['data_adapters'] = adapters
    
    # Setup model configuration
    trading_config['model_type'] = RandomForestModel
    trading_config['model_output_type'] = ModelOutputType.ACTION
    trading_config['model_params'] = config.get('model_params', {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2
    })
    
    # Setup features
    # For now, use a simple default feature set
    # In the future, this should be configurable
    features = []
    if config.get('features'):
        # Parse features from config
        for feature_spec in config['features']:
            if isinstance(feature_spec, list) and len(feature_spec) == 2:
                feature_name, params = feature_spec
                if feature_name == 'rsi':
                    features.append((rsi_feature, params))
                elif feature_name == 'sma':
                    features.append((moving_average_feature, {**params, 'ma_type': 'sma'}))
    else:
        # Default features
        for symbol in trading_config['traded_symbols']:
            features.append((rsi_feature, {'symbol': symbol, 'period': 14}))
            features.append((moving_average_feature, {'symbol': symbol, 'window': 20, 'ma_type': 'sma'}))
    trading_config['features'] = features
    
    # Setup position sizer
    trading_config['position_sizer_type'] = FixedFractionalSizer
    trading_config['position_sizer_params'] = config.get('position_sizer_params', {
        'fraction': 0.1
    })
    
    return trading_config


def parse_datetime(value) -> datetime:
    """Parse various datetime formats"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle ISO format with Z
        value = value.replace('Z', '+00:00')
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # Try common formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
    return datetime(2020, 1, 1)


def extract_backtest_results(engine, config: dict) -> dict:
    """
    Extract results from NautilusTrader backtest engine
    """
    # Get portfolio data from engine - requires Venue argument
    from nautilus_trader.model.identifiers import Venue
    venue = Venue("BACKTESTING")
    account = engine.portfolio.account(venue) if hasattr(engine.portfolio, 'account') else None
    
    starting_capital = float(config['starting_capital'])
    ending_capital = float(account.balance_total().as_decimal()) if account else starting_capital
    total_return = (ending_capital - starting_capital) / starting_capital
    
    # Calculate metrics
    # Note: This is simplified - full implementation would analyze all fills and positions
    trades = []
    equity_curve = []
    
    # Get orders from cache - use orders() instead of fills()
    orders = list(engine.cache.orders()) if hasattr(engine.cache, 'orders') else []
    
    position_map = {}  # symbol -> current position
    
    for order in orders:
        # Only process filled orders
        if hasattr(order, 'filled_qty') and order.filled_qty > 0:
            symbol = order.instrument_id.symbol.value
            side = 'buy' if order.side.value == 'BUY' else 'sell'
            quantity = float(order.filled_qty)
            price = float(order.avg_px) if hasattr(order, 'avg_px') and order.avg_px else 0
            
            # Track position
            if symbol not in position_map:
                position_map[symbol] = 0
            qty_change = quantity if side == 'buy' else -quantity
            position_map[symbol] += qty_change
            
            trades.append({
                'trade_id': str(order.client_order_id),
                'backtest_id': '',  # Will be set by caller
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'timestamp': pd.Timestamp(order.ts_init, unit='ns').to_pydatetime() if hasattr(order, 'ts_init') else None,
                'pnl': None,
                'pnl_percent': None
            })
    
    # Build equity curve from portfolio snapshots if available
    # For now, create a simple curve based on start/end
    days = (config['backtest_end'] - config['backtest_start']).days
    current_equity = starting_capital
    
    for i in range(min(days, 1000)):
        date = config['backtest_start'] + timedelta(days=i)
        # Linear interpolation for demo (would use actual equity snapshots)
        progress = i / days if days > 0 else 1
        current_equity = starting_capital + (ending_capital - starting_capital) * progress
        
        equity_curve.append({
            'date': date.isoformat(),
            'equity': current_equity,
            'cash': current_equity * 0.1,
            'positions': {sym: position_map.get(sym, 0) for sym in config['traded_symbols']}
        })
    
    # Calculate drawdown curve
    drawdown_curve = []
    peak = starting_capital
    for point in equity_curve:
        peak = max(peak, point['equity'])
        drawdown = (peak - point['equity']) / peak if peak > 0 else 0
        drawdown_curve.append({
            'date': point['date'],
            'drawdown': drawdown,
            'peak': peak
        })
    
    # Calculate trade statistics
    winning_trades = sum(1 for t in trades if (t.get('pnl') or 0) > 0)
    losing_trades = len(trades) - winning_trades
    
    return {
        'portfolio': {
            'startingCapital': starting_capital,
            'endingCapital': ending_capital,
            'totalReturn': total_return,
            'annualizedReturn': pow(ending_capital / starting_capital, 1 / max(days / 365, 1)) - 1 if total_return > 0 else total_return,
            'volatility': 0.15,  # Would calculate from returns
            'sharpeRatio': 1.0,  # Would calculate properly
            'sortinoRatio': 1.2,
            'maxDrawdown': max(d['drawdown'] for d in drawdown_curve) if drawdown_curve else 0,
            'calmarRatio': 0.5,
            'winRate': winning_trades / len(trades) if trades else 0,
            'profitFactor': 1.1,
            'expectancy': 100,
            'totalTrades': len(trades),
            'winningTrades': winning_trades,
            'losingTrades': losing_trades
        },
        'risk': {
            'valueAtRisk95': -0.05,
            'expectedShortfall95': -0.08,
            'beta': 1.0,
            'alpha': 0,
            'trackingError': 0.02,
            'informationRatio': 0.5
        },
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'trades': trades
    }

# ============================================================================
# News Endpoints
# ============================================================================

def transform_article_for_api(article: dict) -> dict:
    """
    Transform article from database format to API format.
    Converts source_name to source: {name} for frontend compatibility.
    Also converts url_to_image to urlToImage.
    Converts category string to array.
    """
    transformed = dict(article)
    
    if "source_name" in transformed and transformed["source_name"]:
        transformed["source"] = {"name": transformed["source_name"]}
        del transformed["source_name"]
    
    if "url_to_image" in transformed:
        transformed["urlToImage"] = transformed["url_to_image"]
        del transformed["url_to_image"]
    
    if "category" in transformed and transformed["category"]:
        cat = transformed["category"]
        if isinstance(cat, str):
            transformed["category"] = [cat]
    
    if "tags" not in transformed:
        transformed["tags"] = []
    
    return transformed


@require_auth
async def handle_get_news(request):
    """
    GET /news
    Get financial news articles with pagination
    
    Query parameters:
    - category: News category (general, technology, crypto, forex, politics, developing_markets)
    - page: Page number (default: 1)
    - page_size: Number of articles per page (default: 20)
    """
    try:
        category = request.query.get("category", "general")
        page = int(request.query.get("page", "1"))
        page_size = int(request.query.get("page_size", "20"))
        
        category_map = {
            "general": "general",
            "technology": "technology",
            "crypto": "crypto",
            "forex": "forex",
            "politics": "politics",
            "developing_markets": "developing markets"
        }
        
        internal_category = category_map.get(category, "general")
        
        result = await db.get_news_articles_paginated(
            category=internal_category,
            page=page,
            page_size=page_size
        )
        
        articles = result["articles"]
        pagination = result["pagination"]
        
        articles = [transform_article_for_api(a) for a in articles]
        articles = [serialize_for_json(a) for a in articles]
        
        return web.json_response({
            "category": category,
            "articles": articles,
            "pagination": pagination
        })
    
    except Exception as e:
        print(f"Error in handle_get_news: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_latest_news(request):
    """
    GET /news/latest
    Get latest financial headlines across all categories
    """
    try:
        max_results = int(request.query.get("max_results", "20"))
        
        articles = await news_fetcher.get_latest_headlines(max_results)
        
        articles = [transform_article_for_api(a) for a in articles]
        articles = [serialize_for_json(a) for a in articles]
        
        return web.json_response({
            "articles": articles,
            "count": len(articles)
        })
    
    except Exception as e:
        print(f"Error in handle_get_latest_news: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


@require_auth
async def handle_get_popular_tags(request):
    """
    GET /news/tags
    Get popular tags with article counts
    """
    try:
        limit = int(request.query.get("limit", "20"))
        
        tags = await db.get_popular_tags(limit)
        
        return web.json_response({
            "tags": tags
        })
    
    except Exception as e:
        print(f"Error in handle_get_popular_tags: {e}")
        traceback.print_exc()
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )


async def handle_health(request):
    """Health check endpoint"""
    return web.json_response({"status": "ok"})


async def on_startup(app):
    """Initialize database on startup"""
    await db.initialize()
    print("Database initialized")
    
    # Initialize news fetcher with database for caching
    global news_fetcher
    news_fetcher = get_news_fetcher(db)
    print("News fetcher initialized")


async def on_cleanup(app):
    """Cleanup on shutdown"""
    await db.close()
    print("Database connection closed")


@web.middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses"""
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "*"),
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            }
        )
    
    # Process the request
    response = await handler(request)
    
    # Add CORS headers to response
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


def create_app():
    """Create and configure the aiohttp application"""
    app = web.Application(middlewares=[cors_middleware])

    # Add routes
    app.router.add_post("/lm/generate", handle_generate)
    app.router.add_get("/lm/gethistory", handle_get_history)
    app.router.add_get("/lm/chats", handle_get_chats)
    app.router.add_delete("/lm/deletechat", handle_delete_chat)

    app.router.add_get("/data/tickerdata", handle_ticker_data)
    app.router.add_get("/data/portfolio", handle_portfolio_data)
    app.router.add_post("/data/portfolio/remove", handle_remove_broker)

    # Strategy routes
    app.router.add_get("/strategies", handle_list_strategies)
    app.router.add_post("/strategies", handle_create_strategy)
    app.router.add_get("/strategies/{strategy_id}", handle_get_strategy)
    app.router.add_put("/strategies/{strategy_id}", handle_update_strategy)
    app.router.add_delete("/strategies/{strategy_id}", handle_delete_strategy)
    app.router.add_post("/strategies/{strategy_id}/backtest", handle_run_backtest)
    app.router.add_get("/backtest/{backtest_id}/status", handle_get_backtest_status)
    app.router.add_get("/backtest/{backtest_id}/results", handle_get_backtest_results)

    # News routes
    app.router.add_get("/news", handle_get_news)
    app.router.add_get("/news/latest", handle_get_latest_news)
    app.router.add_get("/news/tags", handle_get_popular_tags)

    app.router.add_get("/health", handle_health)
    
    # Add startup/cleanup handlers
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    return app


async def main():
    """Run the server with SSL"""
    cert_path = pathlib.Path(__file__).with_name("cert.pem")
    key_path = pathlib.Path(__file__).with_name("key.pem")
    
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_path, key_path)
    
    # Create app
    app = create_app()
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "0.0.0.0", 443, ssl_context=ssl_context)
    await site.start()
    
    print("SSE Server started on https://0.0.0.0:443")
    print("Endpoints:")
    print("  POST /lm/generate")
    print("  GET  /lm/gethistory?chatid=<id>")
    print("  GET  /lm/chats")
    print("  GET  /lm/deletechat?chatid=<id>")
    print("  GET  /data/tickerdata")
    print("  GET  /data/portfolio?refresh=true")
    print("  POST /data/portfolio/remove")
    print("  GET  /strategies")
    print("  POST /strategies")
    print("  GET  /strategies/{strategy_id}")
    print("  PUT  /strategies/{strategy_id}")
    print("  DELETE /strategies/{strategy_id}")
    print("  POST /strategies/{strategy_id}/backtest")
    print("  GET  /backtest/{backtest_id}/status")
    print("  GET  /backtest/{backtest_id}/results")
    print("  GET  /health")
    
    # Keep server running
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("Error: cert.pem and key.pem not found.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
