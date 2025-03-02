from collections import defaultdict
import logging
from typing import Any

from fastapi import WebSocket

from app.utils.engine import SupplyChainEngine

class SupplyNetworkManager:
    """Manage multiple supply chain networks"""

    def __init__(self):
        self.networks: dict[str, SupplyChainEngine] = defaultdict(SupplyChainEngine)
        self.network_data: dict[str, dict[str, Any]] = defaultdict(dict)
        self.logger = logging.getLogger(f"{__name__}.SupplyNetworkManager")
        self.logger.info("Initializing supply chain network manager")

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
                self.logger.info("[WS]: Received event 'patch:network:data'")
                await self.handle_patch_data(engine, websocket, network_id, payload)
            elif event == "get:network:actions":
                self.logger.info("[WS]: Received event 'patch:network:data'")
                await self.handle_get_actions(engine, websocket, network_id)
            elif event == "get:network:predictions":
                self.logger.info("[WS]: Received event 'patch:network:data'")
                await self.handle_get_predictions(engine, websocket, network_id)
            else:
                self.logger.warning(f"[WS]: Received unknown event '{event}'")
                await websocket.send_json(
                    {"status": "error", "message": f"Unknown event: {event}"}
                )
        except Exception as e:
            self.logger.error(f"[WS]: unknown error'{str(e)}'")
            await websocket.send_json({"status": "error", "message": str(e)})

    async def handle_patch_data(
    self,
    engine: SupplyChainEngine,
    websocket: WebSocket,
    network_id: str,
    payload: dict[str, Any] | None,
) -> None:
        """Update network with new supplier data"""
        if not payload:
            raise ValueError("No actor payload provided")

        # Check if this is initial external data
        is_initial_data = not engine.external_data_initialized

        # Update engine with latest supplier data
        updated_env = engine.update_environment_with_supplier_data(payload)

        # Special handling for first external data payload
        if is_initial_data:
            self.logger.info(f"Initialized network {network_id} with external data")
            # Reset any existing agent models
            if engine.agent:
                engine.agent = None
            # Re-train prediction models with new data
            updated_env.train_prediction_models()

        await websocket.send_json({
            "status": "success",
            "network_id": network_id,
            "event": "network:data_updated",
            "data_initialized": is_initial_data,
            "message": "Initialized with external data" if is_initial_data else "Appended new external data"
        })

    async def handle_get_actions(
        self, engine: SupplyChainEngine, websocket: WebSocket, network_id: str
    ) -> None:
        """Get optimized actions for the network"""

        agent = engine.load_agent()

        if not agent:
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
