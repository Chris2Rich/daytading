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

SENTINEL = "_____"
ALLOWED_ORIGINS = ["https://zecant.com"]

global_lm = LM_WRAPPER(routers["GROQ"]["models"]["gptoss120"])

clients = {}
active_requests = {} 

def sanitize_error(code: str, message: str = None) -> str:
    """Return a JSON string with a safe error message for the client."""
    error_messages = {
        "TIMEOUT_PER_chunk": f"No chunk received for {PER_CHUNK_TIMEOUT}s",
        "UNKNOWN_ERROR": "An unexpected server error occurred",
        "BUSY": "A previous request is still in progress"
    }
    payload = {
        "type": "error",
        "code": code,
        "data": message or error_messages.get(code, "Unknown error")
    }
    return SENTINEL + "err" + json.dumps(payload)

async def testllm_handler(websocket, client_id, path):
    lm = global_lm

    client_data = clients[client_id]
    if "memory" not in client_data:
        client_data["memory"] = []  # optional per-client memory
    
    try:
        async for message in websocket:

            if active_requests.get(client_id):
                continue

            client_data["memory"].append({"role": "user", "content": message})
            async def stream_with_heartbeat():
                with dspy.context():
                    async for chunk in lm.stream_async_memory(client_data["memory"][-6:], client_data["memory"]):
                        await websocket.send(chunk)

            await websocket.send(f"{SENTINEL}start")
            task = asyncio.create_task(stream_with_heartbeat())
            active_requests[client_id] = task

            # Wrap the task execution in a try...except...finally block
            try:
                await task
            except asyncio.TimeoutError as e:
                if "No chunk received" in str(e):
                    await websocket.send(sanitize_error("TIMEOUT_PER_chunk"))

            except Exception as e:
                # Server-side logging only
                error_id = str(uuid.uuid4())
                try:
                    await websocket.send(sanitize_error("UNKNOWN_ERROR", f"Error ID: {error_id}"))
                    traceback.print_exc()
                except:
                    pass

            finally:
                # Always signal end of stream
                try:
                    await websocket.send(f"{SENTINEL}end")
                except websockets.exceptions.ConnectionClosed:
                    pass
                active_requests[client_id] = None
        
        if active_requests.get(client_id):
            try:
                await active_requests[client_id]
            except asyncio.CancelledError:
                pass
            finally:
                active_requests[client_id] = None

    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        active_requests.pop(client_id, None)

ENDPOINTS = {
    "/testllm": testllm_handler,
}

async def main_handler(websocket):
    path = websocket.request.path
    handler = ENDPOINTS.get(path)
    if handler:
        client_id = str(uuid.uuid4())
        clients[client_id] = {
            "websocket": websocket,
            "lm": None
        }
        try:
            await handler(websocket, client_id, path)
        finally:
            clients.pop(client_id, None)
    else:
        await websocket.close(1011, f"No handler for path: {path}")


async def origin_check(path, request_headers):
    return None
    origin = request_headers.get("Origin")
    if not (origin in ALLOWED_ORIGINS):
        return 403, [], b"Forbidden\n"
    return None

async def main():
    cert_path = pathlib.Path(__file__).with_name("cert.pem")
    key_path = pathlib.Path(__file__).with_name("key.pem")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_path, key_path)

    async with websockets.serve(main_handler, "0.0.0.0", 443, ssl=ssl_context, process_request=origin_check):
        print("Secure WebSocket server started on wss://0.0.0.0:443")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")
    except FileNotFoundError:
        print("Error: cert.pem and key.pem not found.")