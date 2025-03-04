import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas.engine import EngineDataMessage
from app.utils.network import SupplyNetworkManager


class WebsocketConnectionManager:
    """Manager for WebSocket connections with network support"""

    def __init__(self) -> None:
        """
        Initializes an in-memory websocket connection manager
        """
        self.active_connections: list[WebSocket] = []
        self.network_manager = SupplyNetworkManager()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def handle_websocket(self, websocket: WebSocket) -> None:
        try:
            while True:
                try:
                    message = await websocket.receive_text()
                    raw: dict[str, Any] = json.loads(message)
                    event = raw.get("event")
                    network = raw.get("network")
                    payload = raw.get("payload", None)
                except (json.JSONDecodeError, KeyError):
                    await websocket.send_text(
                        json.dumps({"status": "error", "message": "Invalid JSON format"})
                    )
                
                data = {
                    "event": event,
                    "network": network,
                    "payload": payload
                } 
                message = EngineDataMessage(**data)
                await self.network_manager.handle_message(websocket, message)
        except WebSocketDisconnect:
            self.disconnect(websocket)
