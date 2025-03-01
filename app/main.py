import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.utils.engine import SupplyChainEngine
from app.utils.websockets import WSConnectionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SupplyChainEngine()
    
    # Try loading environment
    if not engine.load_environment():
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


wsmanager = WSConnectionManager()

app = FastAPI(title="Meridian Engine", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.websocket("/ws")
async def read_root(ws: WebSocket):
    await wsmanager.connect(ws)
    try:
        while True:
            # Receive supplier data from WebSocket
            raw_data = await ws.receive_text()
            supplier_data = json.loads(raw_data)

            # Update environment with new supplier data
            engine.update_environment_with_supplier_data(supplier_data=supplier_data)  # type: ignore

            # Run optimization with updated data
            top_actions_rewards = engine.optimize()  # type: ignore

            # Generate explanations for the top actions
            explained_actions = []
            for action, reward in top_actions_rewards:
                explanation = engine.explain_action(action, reward)  # type: ignore
                explained_actions.append(explanation)

            # Send back recommended actions
            await wsmanager.send_message(
                json.dumps([exp.dict() for exp in explained_actions]), ws
            )

    except WebSocketDisconnect:
        wsmanager.disconnect(ws)
        await wsmanager.broadcast(
            json.dumps(
                {
                    "message": "A client has disconnected",
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
            )
        )
