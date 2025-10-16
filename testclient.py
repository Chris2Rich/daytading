import asyncio
import ssl
import websockets
import pathlib

async def connect_to_endpoint(uri):
    cert_path = pathlib.Path(__file__).with_name("cert.pem")
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.load_verify_locations(cert_path)
    ssl_context.check_hostname = False # Not recommended for production

    try:
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            while True:
                message = input("Input Message:\t")
                await websocket.send(message)
                print(f"> Sent to {uri}: {message}")

                response = await websocket.recv()
                print(f"< Received from {uri}: {response}")
    except ConnectionRefusedError:
        print(f"Connection to {uri} refused. Is the server running?")
    except Exception as e:
        print(f"An error occurred with {uri}: {e}")

async def main():
    await connect_to_endpoint("wss://localhost:8765/chat")

if __name__ == "__main__":
    asyncio.run(main())