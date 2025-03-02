from typing import List
from pydantic import BaseModel


class Action(BaseModel):
    """
    Schema for action on the supply chain engine 
    """

    supplier_idx: int
    order_quantity: int
    transport_idx: int
    route_idx: int
    production_adjustment: int


class ActionExplanation(BaseModel):
    """
    Schema for action recommendation explanation 
    """

    action: List[int]
    supplier: str
    order_quantity: float
    transport_mode: str
    route: str
    production_adjustment: str
    explanation: str