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

PER_CHUNK_TIMEOUT = 30
MAX_MEMORY_MESSAGES = 50

SENTINEL = "_____"
ALLOWED_ORIGINS = ["https://zecant.com"]

lm = LM_WRAPPER(routers["GROQ"]["models"]["gptoss120"])

clients = {}
active_requests = {} 

def sanitize_error(code: str, message: str = None) -> str:
    """Return a JSON string with a safe error message for the client."""
    error_messages = {
        "TIMEOUT_PER_CHUNK": f"No chunk received for {PER_CHUNK_TIMEOUT}s",
        "UNKNOWN_ERROR": "An unexpected server error occurred",
        "BUSY": "A previous request is still in progress"
    }
    payload = {
        "type": "error",
        "code": code,
        "data": message or error_messages.get(code, "Unknown error")
    }
    return SENTINEL + "err" + json.dumps(payload)

async def rawllm_handler(websocket, client_id, path):
    client_data = clients[client_id]
    if "memory" not in client_data:
        client_data["memory"] = []
    
    try:
        async for message in websocket:
            if active_requests.get(client_id):
                await websocket.send(sanitize_error("BUSY"))
                continue

            if len(client_data["memory"]) > MAX_MEMORY_MESSAGES:
                client_data["memory"] = client_data["memory"][-MAX_MEMORY_MESSAGES:]
            
            client_data["memory"].append({"role": "user", "content": message})
            
            async def stream_with_timeout():
                """Stream with per-chunk timeout monitoring"""
                last_chunk_time = asyncio.get_event_loop().time()
                
                async for chunk in lm.stream_async_memory(
                    client_data["memory"][-6:], 
                    client_data["memory"]
                ):
                    await websocket.send(chunk)
                    last_chunk_time = asyncio.get_event_loop().time()
                
            await websocket.send(f"{SENTINEL}start")
            
            try:
                task = asyncio.create_task(stream_with_timeout())
                active_requests[client_id] = task
                
                await asyncio.wait_for(task, timeout=PER_CHUNK_TIMEOUT * 10)
                
            except asyncio.TimeoutError:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                await websocket.send(sanitize_error("TIMEOUT_PER_CHUNK"))
                
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
                
                try:
                    await websocket.send(sanitize_error("UNKNOWN_ERROR", f"Error ID: {error_id}"))
                except:
                    pass
                
                if client_data["memory"] and client_data["memory"][-1]["role"] == "assistant":
                    client_data["memory"].pop()

            finally:
                try:
                    await websocket.send(f"{SENTINEL}end")
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    active_requests[client_id] = None
        
    except websockets.exceptions.ConnectionClosedError:
        pass
    except websockets.exceptions.ConnectionClosedOK:
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


async def origin_check(path, request_headers):
    """Enable origin checking for production security"""
    # return None
    
    origin = request_headers.get("Origin")
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
        ping_interval=20,
        ping_timeout=10,
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