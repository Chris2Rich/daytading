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
import aiohttp_cors

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from llm.lmwrapper import LM_WRAPPER_WITH_TOOLS, routers
from db.database import Database

# Clerk configuration
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
if not CLERK_SECRET_KEY:
    raise ValueError("CLERK_SECRET_KEY environment variable is required")

# Initialize Clerk SDK
clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)

# Initialize LLM and Database
lm = LM_WRAPPER_WITH_TOOLS(routers["GROQ"]["models"]["gptoss120"])
db = Database()


async def verify_clerk_token(token: str) -> Optional[str]:
    """
    Verify Clerk JWT token using official SDK and return clerk_user_id if valid.
    Returns None if invalid.
    """
    try:
        # Verify the session token using Clerk's verify_token method
        request_state = await asyncio.to_thread(
            clerk.verify_token,
            token
        )
        
        # Extract user ID from the verified token
        # The SDK returns a RequestState object with claims
        if request_state:
            # Try to get user_id from claims (sub claim contains the user ID)
            if hasattr(request_state, 'sub'):
                return request_state.sub
            elif hasattr(request_state, 'claims'):
                return request_state.claims.get('sub')
        
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
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        clerk_user_id = await verify_clerk_token(token)
        
        if not clerk_user_id:
            return web.json_response(
                {"error": "Invalid token"},
                status=401
            )
        
        # Add clerk_user_id to request for use in handler
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
        
        # Create new chat if no chat_id provided
        if not chat_id:
            chat_id = await db.create_chat(clerk_user_id)
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
        
        # Stream LLM response
        assistant_response = ""
        try:
            async for token_data in lm.stream_with_tools_execution(
                user_message=message,
                chat_history=chat_history[:-1],  # Exclude the message we just added
                requested_tools=tools
            ):
                if token_data["type"] == "token":
                    token = token_data["content"]
                    assistant_response += token
                    
                    # Send token via SSE
                    await response.write(
                        f"data: {json.dumps({'type': 'token', 'content': token})}\n\n".encode()
                    )
                elif token_data["type"] == "tool_call":
                    # Future: Handle tool calls
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
        
        # Convert datetime objects to ISO format strings
        for msg in history:
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
        
        # Convert datetime objects to ISO format strings
        for chat in chats:
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


def create_app():
    """Create and configure the aiohttp application"""
    app = web.Application()
    
    cors = aiohttp_cors.setup(app, defaults={
        "https://zecant.com": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        ),
        "https://www.zecant.com": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        ),
        "http://localhost:3000": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        ),
    })

    # Add routes
    app.router.add_post("/lm/generate", handle_generate)
    app.router.add_get("/lm/gethistory", handle_get_history)
    app.router.add_get("/lm/chats", handle_get_chats)
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
