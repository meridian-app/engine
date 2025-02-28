from typing import Any, Optional
import gymnasium as gym
import numpy as np
import pandas as pd

from app.schemas.action import ActionExplanation
from app.environments.engine import SupplyChainEngine

class SupplyChainEngineOptimizer:
    """Class to handle supply chain optimization and explanation."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the optimizer with a supply chain environment."""
        self.env = SupplyChainEngine(csv_path=csv_path, render_mode=None)
        self.reset_environment()

    def reset_environment(self):
        """Reset the environment to initial state."""
        observation, _ = self.env.reset()
        self.current_observation = observation
        return observation
    
    def explain_action(self, action, reward):
        """Generate a human-readable explanation for an action."""
        supplier_idx, order_qty_scaled, transport_idx, route_idx, prod_vol_adj = action
        
        # Convert scaled order quantity to actual value
        min_order = 10
        max_order = 100
        order_quantity = min_order + (max_order - min_order) * (order_qty_scaled / 10.0)
        
        # Get the named values for categorical variables
        supplier = self.env.suppliers[supplier_idx]
        transport_mode = self.env.transport_modes[transport_idx]
        route = self.env.routes[route_idx]
        
        # Convert production volume adjustment to a description
        prod_adj_percentages = [-0.2, -0.1, 0, 0.1, 0.2]  # -20% to +20%
        prod_vol_change = prod_adj_percentages[prod_vol_adj]
        
        if prod_vol_change < 0:
            prod_adjustment = f"Decrease by {abs(prod_vol_change*100)}%"
        elif prod_vol_change > 0:
            prod_adjustment = f"Increase by {prod_vol_change*100}%"
        else:
            prod_adjustment = "No change"
        
        # Create explanation text
        explanation = (
            f"Based on current supply chain conditions, we recommend working with {supplier} "
            f"to place an order of {order_quantity:.1f} units. "
            f"Use {transport_mode} transportation through {route}. "
            f"For production volumes, {prod_adjustment.lower()}. "
            f"optimizing the balance between profit, lead time, and quality."
        )
        
        # Create detailed explanation with reasoning
        detailed_explanation = ActionExplanation(
            action=action.tolist(),
            supplier=supplier,
            order_quantity=order_quantity,
            transport_mode=transport_mode,
            route=route,
            production_adjustment=prod_adjustment,
            expected_reward=reward,
            explanation=explanation
        )
        
        return detailed_explanation

    def update_environment_with_supplier_data(self, supplier_data: dict[str, Any]):
        """Update the environment based on real-time supplier data."""
        # Extract supplier information
        supplier_name = supplier_data.get("name", "")
        
        # Check if this supplier exists in our environment
        if supplier_name in self.env.suppliers:
            self.env.suppliers.index(supplier_name)
        else:
            # Add new supplier to the environment
            self.env.suppliers.append(supplier_name)
        
        # Update the environment's data with this supplier's information
        # Extract product information from the first product (assuming there's at least one)
        if supplier_data.get("products") and len(supplier_data["products"]) > 0:
            product = supplier_data["products"][0]
            
            # Create a new row for the data
            new_data = {
                "Price": product.get("price", 0),
                "Stock levels": product.get("stock_level", 0),
                "Lead time": product.get("manufacturing_details", {}).get("lead_time", 0),
                "Production volumes": product.get("manufacturing_details", {}).get("production_volume", 0),
                "Manufacturing lead time": product.get("manufacturing_details", {}).get("manufacturing_lead_time", 0),
                "Manufacturing costs": product.get("manufacturing_details", {}).get("manufacturing_costs", 0),
                "Defect rates": product.get("manufacturing_details", {}).get("defect_rates", 0),
                "Supplier name": supplier_name,
            }
            
            # Handle shipping options
            if product.get("shipping_options") and len(product["shipping_options"]) > 0:
                shipping = product["shipping_options"][0]
                new_data["Shipping costs"] = shipping.get("cost", 0)
                
                # Update transportation modes if needed
                transport_mode = shipping.get("mode", "")
                if transport_mode and transport_mode not in self.env.transport_modes:
                    self.env.transport_modes.append(transport_mode) # 
                
                # Update routes if needed
                route = shipping.get("route", "")
                if route and route not in self.env.routes:
                    self.env.routes.append(route)
                
                # Set values
                new_data["Transportation modes"] = transport_mode
                new_data["Routes"] = route
            
            # Calculate costs (simple sum for now)
            new_data["Costs"] = new_data.get("Manufacturing costs", 0) + new_data.get("Shipping costs", 0)
            
            # Calculate revenue
            effective_production = new_data.get("Production volumes", 0) * (1 - new_data.get("Defect rates", 0) / 100)
            new_data["Revenue generated"] = effective_production * new_data.get("Price", 0)
            
            # Add this data to the environment's dataset
            new_df = pd.DataFrame([new_data])
            self.env.full_data = pd.concat([self.env.full_data, new_df], ignore_index=True)
            
        # Update environment action space if needed
        num_suppliers = len(self.env.suppliers)
        num_transport_modes = len(self.env.transport_modes)
        num_routes = len(self.env.routes)
        
        self.env.action_space = gym.spaces.MultiDiscrete(
            [
                num_suppliers,          # Supplier selection
                10,                     # Order quantity (scaled 1-10)
                num_transport_modes,    # Transportation mode
                num_routes,             # Route selection
                5,                      # Production volume adjustment
            ]
        )
        
        # Update observation space
        num_features = 9  # numerical features
        obs_dim = num_features + num_suppliers + num_transport_modes + num_routes
        self.env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        return self.env

    def monte_carlo_optimization(self, num_simulations=100, horizon=10):
        """
        Use Monte Carlo simulation to find optimal actions.
        
        Args:
            env: The supply chain environment
            num_simulations: Number of Monte Carlo simulations 
            horizon: Planning horizon (steps to simulate)
            
        Returns:
            The best actions found and their expected rewards
        """
        # Store top 3 best actions and rewards
        top_actions = []
        top_rewards = []
        
        # Get current state
        current_observation = self.env._get_state_representation()
        
        # Run multiple simulations to find the best actions
        for _ in range(num_simulations):
            # Sample a random action
            action = self.env.action_space.sample()
            
            # Create a copy of the current environment state
            test_env = SupplyChainEngine(render_mode=None)
            test_env.reset()
            test_env.current_state = self.env.current_state.copy()
            test_env.current_supplier_idx = self.env.current_supplier_idx
            test_env.current_transport_idx = self.env.current_transport_idx
            test_env.current_route_idx = self.env.current_route_idx
            
            # Apply the action
            _, reward, _, _, _ = test_env.step(action)
            total_reward = reward
            
            # Simulate future steps with random actions
            for _ in range(horizon - 1):
                future_action = test_env.action_space.sample()
                _, reward, done, _, _ = test_env.step(future_action)
                total_reward += reward * 0.9  # Apply discount factor
                if done:
                    break
            
            # Check if this is among the top actions
            if len(top_actions) < 3 or total_reward > min(top_rewards):
                # Add this action and reward
                top_actions.append(action)
                top_rewards.append(total_reward)
                
                # Sort and keep only the top 3
                if len(top_actions) > 3:
                    # Find index of minimum reward
                    min_idx = top_rewards.index(min(top_rewards))
                    # Remove the action with the minimum reward
                    top_actions.pop(min_idx)
                    top_rewards.pop(min_idx)
        
        return list(zip(top_actions, top_rewards))
    # def optimize(self, num_simulations=100, horizon=5):
    #     """
    #     Monte Carlo simulation to find optimal actions.
        
    #     Args:
    #         num_simulations: Number of simulations to run
    #         horizon: Planning horizon (number of steps to look ahead)
            
    #     Returns:
    #         The best action found and its expected reward
    #     """
    #     best_reward = float("-inf")
    #     best_action = None

    #     # Try different possible actions
    #     for _ in range(num_simulations):
    #         # Sample a random action
    #         action = self.env.action_space.sample()
            
    #         # Make a copy of the current environment state
    #         test_env = SupplyChainEngine(render_mode=None)
    #         test_env.reset()
            
    #         # Apply the action
    #         _, reward, _, _, _ = test_env.step(action)
    #         total_reward = reward
            
    #         # Simulate future steps with random actions
    #         for _ in range(horizon - 1):
    #             future_action = test_env.action_space.sample()
    #             _, reward, done, _, _ = test_env.step(future_action)
    #             total_reward += reward
    #             if done:
    #                 break
                    
    #         # Update if this is the best action so far
    #         if total_reward > best_reward:
    #             best_reward = total_reward
    #             best_action = action
                
    #     return best_action, best_reward

