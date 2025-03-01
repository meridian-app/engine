import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation

from app.constants import AGENT_MODEL_PATH, ENVIRONMENT_MODEL_PATH, MODELS_DIR
from app.engine.agent import SupplyChainAgent
from app.engine.environment import SupplyChainEnvironment
from app.schemas.action import ActionExplanation


class SupplyChainEngine:
    """Wrapper class to handle supply chain optimization, action explanation and agent based tasks."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the optimizer with a supply chain environment."""
        self.env = SupplyChainEnvironment(csv_path=csv_path, render_mode=None)
        self.agent: Optional[SupplyChainAgent] = None
        self.reset_environment()

    def pre_train_environment(self, num_simulations=1000):
        """Pre-train environment using Monte Carlo simulations"""
        print("Pre-training environment...")
        self.train_environment(num_simulations=num_simulations)
        print("\nEnvironment pre-training complete!")

    def save_environment(self):
        """Save environment state"""
        import os
        import pickle
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agent_path = os.path.join(current_dir, "..", ENVIRONMENT_MODEL_PATH)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.join(current_dir, "..", MODELS_DIR), exist_ok=True)
        with open(agent_path, "wb+") as f:
            pickle.dump(self.env, f)
        print(f"Environment saved to {ENVIRONMENT_MODEL_PATH}")

    def load_environment(self) -> bool:
        """Load environment state if available"""
        import pickle

        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, "..", ENVIRONMENT_MODEL_PATH)

        if Path(env_path).exists():
            with open(env_path, "rb") as f:
                self.env = pickle.load(f)
            print(f"Environment loaded from {env_path}")
            return True
        return False

    def reset_environment(self):
        """Reset the environment to initial state."""
        observation, _ = self.env.reset()
        self.current_observation = observation
        return observation

    def save_agent(self):
        """Save trained agent"""
        if self.agent:
            # Create the directory if it doesn't exist
            import os
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            agent_path = os.path.join(current_dir, "..", AGENT_MODEL_PATH)

            os.makedirs(os.path.join(current_dir, "..", MODELS_DIR), exist_ok=True)
            self.agent.save_model(agent_path)
            print(f"Agent saved to {agent_path}")

    def load_agent(self) -> bool:
        """Load trained agent if available"""
        import os
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agent_path = os.path.join(current_dir, "..", AGENT_MODEL_PATH)

        if Path(agent_path).exists():
            if not self.agent:
                self.agent = SupplyChainAgent(self.env)
            self.agent.load_model(agent_path)
            print(f"Agent loaded from {agent_path}")
            return True
        return False

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
            explanation=explanation,
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
                "Lead time": product.get("manufacturing_details", {}).get(
                    "lead_time", 0
                ),
                "Production volumes": product.get("manufacturing_details", {}).get(
                    "production_volume", 0
                ),
                "Manufacturing lead time": product.get("manufacturing_details", {}).get(
                    "manufacturing_lead_time", 0
                ),
                "Manufacturing costs": product.get("manufacturing_details", {}).get(
                    "manufacturing_costs", 0
                ),
                "Defect rates": product.get("manufacturing_details", {}).get(
                    "defect_rates", 0
                ),
                "Supplier name": supplier_name,
            }

            # Handle shipping options
            if product.get("shipping_options") and len(product["shipping_options"]) > 0:
                shipping = product["shipping_options"][0]
                new_data["Shipping costs"] = shipping.get("cost", 0)

                # Update transportation modes if needed
                transport_mode = shipping.get("mode", "")
                if transport_mode and transport_mode not in self.env.transport_modes:
                    self.env.transport_modes.append(transport_mode)  #

                # Update routes if needed
                route = shipping.get("route", "")
                if route and route not in self.env.routes:
                    self.env.routes.append(route)

                # Set values
                new_data["Transportation modes"] = transport_mode
                new_data["Routes"] = route

            # Calculate costs (simple sum for now)
            new_data["Costs"] = new_data.get("Manufacturing costs", 0) + new_data.get(
                "Shipping costs", 0
            )

            # Calculate revenue
            effective_production = new_data.get("Production volumes", 0) * (
                1 - new_data.get("Defect rates", 0) / 100
            )
            new_data["Revenue generated"] = effective_production * new_data.get(
                "Price", 0
            )

            # Add this data to the environment's dataset
            new_df = pd.DataFrame([new_data])
            self.env.full_data = pd.concat(
                [self.env.full_data, new_df], ignore_index=True
            )

        # Update environment action space if needed
        num_suppliers = len(self.env.suppliers)
        num_transport_modes = len(self.env.transport_modes)
        num_routes = len(self.env.routes)

        self.env.action_space = gym.spaces.MultiDiscrete(
            [
                num_suppliers,  # Supplier selection
                10,  # Order quantity (scaled 1-10)
                num_transport_modes,  # Transportation mode
                num_routes,  # Route selection
                5,  # Production volume adjustment
            ]
        )

        # Update observation space
        num_features = 9  # numerical features
        obs_dim = num_features + num_suppliers + num_transport_modes + num_routes
        self.env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        return self.env

    def train_environment(self, num_simulations=100, horizon=10):
        """
        Use Monte Carlo simulation to find optimal actions with prediction logging.
        """
        top_actions = []
        top_rewards = []

        current_observation = self.env._get_state_representation()

        for _ in range(num_simulations):
            action = self.env.action_space.sample()
            test_env = self._create_test_env_copy()

            # Apply the initial action and get predictions
            _next_state, reward, _terminated, _truncated, info = test_env.step(action)
            total_reward = reward

            # Print predictions and actuals
            print(f"Reward: {reward:.2f}")
            print(
                f"Lead Time: {info['metrics']['lead_time']:.1f} (Predicted: {info['predictions']['Lead time']:.1f})"
            )
            print(
                f"Costs: {info['metrics']['costs']:.2f} (Predicted: {info['predictions']['Costs']:.2f})"
            )

            # Simulate future steps
            for _ in range(horizon - 1):
                future_action = test_env.action_space.sample()
                _, future_reward, done, _, _ = test_env.step(future_action)
                total_reward += future_reward * 0.9
                if done:
                    break

            # Update top actions
            if len(top_actions) < 3 or total_reward > min(top_rewards):
                top_actions.append(action)
                top_rewards.append(total_reward)
                # Maintain only top 3
                if len(top_actions) > 3:
                    min_idx = top_rewards.index(min(top_rewards))
                    top_actions.pop(min_idx)
                    top_rewards.pop(min_idx)

        return list(zip(top_actions, top_rewards))

    def train_and_evaluate_agent(self, num_episodes=500):
        """Train and evaluate agent with environment pre-training"""
        # Pre-train environment if not already trained
        if not Path(ENVIRONMENT_MODEL_PATH).exists():
            self.pre_train_environment()
            self.save_environment()

        # Load or train agent
        if not self.load_agent():
            print("Training new agent...")
            env = FlattenObservation(self.env)
            self.agent = SupplyChainAgent(env)
            agent_results = self.agent.train(num_episodes=num_episodes)
            self.save_agent()

            # Evaluate and show results
            eval_results = self.agent.evaluate(num_episodes=20)
            self.agent.plot_training_results()

            return agent_results, eval_results

        print("Using pre-trained agent")
        return None, None

    def _create_test_env_copy(self):
        """Helper to create an environment copy"""
        test_env = SupplyChainEnvironment()
        test_env.current_state = self.env.current_state.copy()
        test_env.current_supplier_idx = self.env.current_supplier_idx
        test_env.current_transport_idx = self.env.current_transport_idx
        test_env.current_route_idx = self.env.current_route_idx
        return test_env
