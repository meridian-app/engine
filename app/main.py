import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.utils.engine import SupplyChainEngine
from app.utils.websocket import WebsocketConnectionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SupplyChainEngine()

    check_environment = engine.load_environment()

    # Try loading environment
    if check_environment == False:
        # Pre-train and save environment if not loaded
        engine.pre_train_environment()
        engine.save_environment()

        # Start agent training in background
        import asyncio

        loop = asyncio.get_event_loop()

        async def train_async():
            await loop.run_in_executor(None, engine.train_and_evaluate_agent)

        asyncio.create_task(train_async())

    yield  # Server starts here

    # Cleanup
    if engine:
        engine.env.close()


wsmanager = WebsocketConnectionManager()

app = FastAPI(title="Meridian Engine", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await wsmanager.connect(ws)
    try:
        await wsmanager.handle_websocket(ws)
    except WebSocketDisconnect:
        wsmanager.disconnect(ws)
