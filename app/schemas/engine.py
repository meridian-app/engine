from typing import Dict, List
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
    performance_metrics: Dict
    location: Dict
    capacity: Dict
    certifications: List[str]
