import json

from fastapi import WebSocket, WebSocketDisconnect

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
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    await self.network_manager.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"status": "error", "message": "Invalid JSON format"}
                    )
        except WebSocketDisconnect:
            self.disconnect(websocket)
