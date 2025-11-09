import asyncio
import ssl
import websockets
import pathlib
import uuid
import sys
import os
import traceback
import aiohttp
import json
import dspy

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from llm.lmwrapper import LM_WRAPPER, routers

MAX_MEMORY_MESSAGES = 50

SENTINEL = "_____"
ALLOWED_ORIGINS = [
    "https://zecant.com",
    "https://www.zecant.com",
    "http://localhost:3000",
    "http://localhost:3001",
]

lm = LM_WRAPPER(routers["GROQ"]["models"]["gptoss120"])

clients = {}
active_requests = {} 

def sanitize_error(code: str, message: str = None) -> str:
    """Return a JSON string with a safe error message for the client."""
    error_messages = {
        "TIMEOUT_GENERATION": f"Generation timed out",
        "UNKNOWN_ERROR": "An unexpected server error occurred",
        "BUSY": "A previous request is still in progress"
    }
    payload = {
        "type": "error",
        "code": code,
        "data": message or error_messages.get(code, "Unknown error")
    }
    return SENTINEL + "err" + json.dumps(payload)

async def safe_send(ws, data):
    """
    Attempt to send `data`. If the connection is closed, swallow the exception
    and return False; otherwise return True.
    """
    try:
        await ws.send(data)
        return True
    except (websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosed) as exc:
        return False

async def rawllm_handler(websocket, client_id, path):
    client_data = clients[client_id]
    if "memory" not in client_data:
        client_data["memory"] = []

    try:
        async for message in websocket:

            if active_requests.get(client_id):
                await safe_send(websocket, sanitize_error("BUSY"))
                continue

            if len(client_data["memory"]) > MAX_MEMORY_MESSAGES:
                client_data["memory"] = client_data["memory"][-MAX_MEMORY_MESSAGES:]

            client_data["memory"].append({"role": "user", "content": message})

            async def stream_response():
                try:
                    async for chunk in lm.stream_async_memory(
                        client_data["memory"][-6:],
                        client_data["memory"]
                    ):
                        ok = await safe_send(websocket, chunk)
                        if not ok:
                            break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    traceback.print_exc()

            await safe_send(websocket, f"{SENTINEL}start")

            try:
                task = asyncio.create_task(stream_response())
                active_requests[client_id] = task

                try:
                    await task

                except asyncio.TimeoutError:
                    if not task.done():
                        task.cancel()
                    await safe_send(websocket, sanitize_error("TIMEOUT_GENERATION"))
                    if client_data["memory"] and client_data["memory"][-1]["role"] == "assistant":
                        client_data["memory"].pop()

            except Exception as e:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                error_id = str(uuid.uuid4())
                print(f"Error {error_id}: {type(e).__name__}: {e}")
                traceback.print_exc()

                await safe_send(websocket, sanitize_error("UNKNOWN_ERROR", f"Error ID: {error_id}"))

                if client_data["memory"] and client_data["memory"][-1]["role"] == "assistant":
                    client_data["memory"].pop()

            finally:
                await safe_send(websocket, f"{SENTINEL}end")
                active_requests[client_id] = None

    except (websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosed):
        pass

    finally:
        if client_id in active_requests and active_requests[client_id]:
            task = active_requests[client_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        active_requests.pop(client_id, None)

ENDPOINTS = {
    "/rawllm": {"fn": rawllm_handler, "lm": True},
}

async def main_handler(websocket):
    path = websocket.request.path
    handler_info = ENDPOINTS.get(path)
    if not handler_info:
        await websocket.close(1011, f"No handler for path: {path}")
        return

    client_id = str(uuid.uuid4())
    context = dspy.context() if handler_info.get("lm") else None

    clients[client_id] = {
        "websocket": websocket,
        "context": context,
        "memory": [],
    }

    try:
        if context:
            with context:
                await handler_info["fn"](websocket, client_id, path)
        else:
            await handler_info["fn"](websocket, client_id, path)
    finally:
        clients.pop(client_id, None)
        active_requests.pop(client_id, None)


async def origin_check(path, request):
    """Enable origin checking for production security"""
    # return None
    
    origin = request.headers.get("Origin")
    if not (origin in ALLOWED_ORIGINS):
        return 403, [], b"Forbidden\n"
    return None

async def main():
    cert_path = pathlib.Path(__file__).with_name("cert.pem")
    key_path = pathlib.Path(__file__).with_name("key.pem")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_path, key_path)

    async with websockets.serve(
        main_handler, 
        "0.0.0.0", 
        443, 
        ssl=ssl_context, 
        process_request=origin_check,
        ping_interval=60,
        ping_timeout=30,
        max_size=2**30,
        max_queue=None,
        close_timeout=10,
    ):
        print("Secure WebSocket server started on wss://0.0.0.0:443")
        await asyncio.Future()

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