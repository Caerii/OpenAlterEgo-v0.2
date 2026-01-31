import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8765"
    async with websockets.connect(uri) as ws:
        print("connected:", uri)
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(data)

if __name__ == "__main__":
    asyncio.run(main())
