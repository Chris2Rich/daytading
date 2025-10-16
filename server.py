import asyncio
import ssl
import websockets
import pathlib

async def chat_handler(websocket, path):
    print(f"Client connected to {path}")
    try:
        async for message in websocket:
            print(f"< Received from {path}: {message}")
            response = f"Message: {message}, ID: {websocket.id}"
            await websocket.send(response)
            print(f"> Sent to {path}: {response}")
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client disconnected from {path}")

ENDPOINTS = {
    "/chat": chat_handler,
}

async def main_handler(websocket):
    path = websocket.request.path
    handler = ENDPOINTS.get(path)
    if handler:
        await handler(websocket, path)
    else:
        print(f"Connection rejected for path: {path}")
        await websocket.close(1011, f"No handler for path: {path}")

async def main():
    cert_path = pathlib.Path(__file__).with_name("cert.pem")
    key_path = pathlib.Path(__file__).with_name("key.pem")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_path, key_path)

    async with websockets.serve(main_handler, "localhost", 8765, ssl=ssl_context):
        print("Secure WebSocket server started on wss://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")
    except FileNotFoundError:
        print("Error: cert.pem and key.pem not found. Please generate them first.")