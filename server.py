import asyncio
import ssl
import pathlib
import sys
import os
import traceback
import json
from aiohttp import web
from functools import wraps
from typing import Optional, List, Dict, Any
from clerk_backend_api import Clerk
import jwt
from jwt import PyJWKClient

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from llm.lmwrapper import LM_WRAPPER, LM_WRAPPER_WITH_TOOLS, routers
from db.database import Database
from retrieval.yahoofinance import (
    get_trading, 
    get_option, 
    get_option_chain, 
    get_fundamentals, 
    get_price_targets
)

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
lm = LM_WRAPPER_WITH_TOOLS(routers["GROQ"]["models"]["gptoss120"])
titlelm = LM_WRAPPER(routers["GROQ"]["models"]["gptoss120"])
db = Database()


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
        try:
            async for token_data in lm.stream_with_tools_execution(
                user_message=message,
                chat_history=chat_history[:-1],  # Exclude the message we just added
                requested_tools=tools,
                auto_search=True  # Enable automatic web search
            ):
                if token_data["type"] == "search_performed":
                    # Forward search event to frontend
                    search_event = {
                        "type": "search_performed",
                        "query": token_data.get("query", ""),
                        "result_count": token_data.get("results", {}).get("total", 0) if token_data.get("results") else 0
                    }
                    await response.write(
                        f"data: {json.dumps(search_event)}\n\n".encode()
                    )
                    # Small delay to ensure frontend processes search event before tokens
                    await asyncio.sleep(0.1)
                    
                elif token_data["type"] == "token":
                    token = token_data["content"]
                    assistant_response += token
                    
                    # Send token via SSE
                    await response.write(
                        f"data: {json.dumps({'type': 'token', 'content': token})}\n\n".encode()
                    )
                    
                elif token_data["type"] == "tool_call":
                    # Future: Handle other tool calls
                    await response.write(
                        f"data: {json.dumps(token_data)}\n\n".encode()
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
        
        # Convert datetime and UUID objects to strings
        for msg in history:
            # Convert UUIDs to strings
            if "message_id" in msg:
                msg["message_id"] = str(msg["message_id"])
            if "chat_id" in msg:
                msg["chat_id"] = str(msg["chat_id"])
            
            # Convert datetime to ISO format
            if "timestamp" in msg:
                msg["timestamp"] = msg["timestamp"].isoformat()
        
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
        
        # Convert datetime and UUID objects to strings
        for chat in chats:
            # Convert UUID to string
            if "chat_id" in chat:
                chat["chat_id"] = str(chat["chat_id"])
            
            # Convert datetime objects to ISO format strings
            for key in ["created_at", "updated_at"]:
                if key in chat and chat[key]:
                    chat[key] = chat[key].isoformat()
        
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


async def handle_health(request):
    """Health check endpoint"""
    return web.json_response({"status": "ok"})


async def on_startup(app):
    """Initialize database on startup"""
    await db.initialize()
    print("Database initialized")


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