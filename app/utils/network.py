from collections import defaultdict
from typing import Any

from fastapi import WebSocket

from app.utils.engine import SupplyChainEngine


class SupplyNetworkManager:
    """Manage multiple supply chain networks"""

    def __init__(self):
        self.networks: dict[str, SupplyChainEngine] = defaultdict(SupplyChainEngine)
        self.network_data: dict[str, dict[str, Any]] = defaultdict(dict)

    async def handle_message(self, websocket: WebSocket, message: dict[str, Any]):
        event = message.get("event")
        network_id = message.get("network_id")
        payload = message.get("payload")

        if not event or not network_id:
            await websocket.send_json(
                {"status": "error", "message": "Missing event or network_id"}
            )
            return

        engine = self.networks[network_id]

        try:
            if event == "patch:network:data":
                await self.handle_patch_data(engine, network_id, payload)
            elif event == "get:network:actions":
                await self.handle_get_actions(engine, websocket, network_id)
            elif event == "get:network:predictions":
                await self.handle_get_predictions(engine, websocket, network_id)
            else:
                await websocket.send_json(
                    {"status": "error", "message": f"Unknown event: {event}"}
                )
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})

    async def handle_patch_data(
        self,
        engine: SupplyChainEngine,
        network_id: str,
        payload: dict[str, Any] | None,
    ) -> None:
        """Update network with new supplier data"""

        if not payload:
            raise ValueError("No actor payload provided")

        # Update network-specific data
        self.network_data[network_id].update(payload)

        # Update engine with latest supplier data
        engine.update_environment_with_supplier_data(payload)
        engine.save_environment()

    async def handle_get_actions(
        self, engine: SupplyChainEngine, websocket: WebSocket, network_id: str
    ) -> None:
        """Get optimized actions for the network"""

        if not engine.load_agent():
            await websocket.send_json(
                {
                    "status": "info",
                    "network_id": network_id,
                    "message": "Agent is not trained",
                }
            )
            top_actions = engine.train_environment()
        else:
            observation = engine.reset_environment()

            action = engine.agent.choose_action(observation)  # type: ignore
            top_actions = [(action)]

            explained_actions = [
                engine.explain_action(action) for action in top_actions
            ]

            await websocket.send_json(
                {
                    "status": "success",
                    "network_id": network_id,
                    "event": "network:actions",
                    "data": [exp.model_dump() for exp in explained_actions],
                }
            )

    async def handle_get_predictions(
        self, engine: SupplyChainEngine, websocket: WebSocket, network_id: str
    ) -> None:
        """Get current network predictions"""

        predictions = engine.env.predict_supply_chain_values()

        await websocket.send_json(
            {
                "status": "success",
                "network_id": network_id,
                "event": "network:predictions",
                "data": predictions,
            }
        )
