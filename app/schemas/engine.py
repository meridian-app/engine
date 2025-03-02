from typing import Dict, List, Literal

from pydantic import BaseModel


class EngineMetrics(BaseModel):
    """
    Schema for the metrics output by the supply chain engine
    """

    revenue: float
    costs: float
    lead_time: float
    defect_rate: float
    profit: float
    total_reward: float


class EngineActorData(BaseModel):
    """
    Schema for the actor external data into the supply chain engine
    """

    supplier_id: str
    name: str
    products: List[Dict]
    performance_metrics: Dict | None
    location: Dict | None
    capacity: Dict | None
    certifications: List[str] | None


class EngineDataMessage(BaseModel):
    event: Literal["get:network:actions", "get:network:predictions", "update:network:data"]
    network: str # ID (UUID) of the supply chain network on meridian platform
    payload: EngineActorData | None
