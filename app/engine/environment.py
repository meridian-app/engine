import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.ensemble import RandomForestRegressor


class SupplyChainEnvironment(gym.Env):
    """
    A custom Gymnasium environment for supply chain optimization in manufacturing.

    This environment simulates a supply chain network with multiple suppliers,
    focusing on optimizing key metrics like revenue, costs, lead times, and
    production quality.

    Observation Space:
        - Price
        - Stock levels
        - Lead time
        - Production volumes
        - Manufacturing lead time
        - Manufacturing costs
        - Defect rates
        - Shipping costs
        - Costs (total)
        - Supplier (one-hot encoded)
        - Transportation mode (one-hot encoded)
        - Route (one-hot encoded)

    Action Space:
        - Supplier selection
        - Order quantity
        - Transportation mode
        - Route selection
        - Production volume adjustment
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        csv_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        synthetic_data_size: int = 500,
        simulation_steps: int = 30,
        random_disruptions: bool = True,
        lead_time_weight: float = 0.5,  # Increased weight for lead time in reward calculation
    ):
        super().__init__()

        self.logger = logging.getLogger(f"{__name__}.SupplyChainEnvironment")
        self.logger.info("Initializing supply chain environment")

        # Set default csv_path if none provided
        if csv_path is None:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to app directory, then to data directory
            csv_path = os.path.join(current_dir, "..", "data", "supply_chain_data.csv")

        # Load and preprocess data
        self.base_data = self._load_and_preprocess_data(csv_path)

        # Generate synthetic data for more robust training
        self.full_data = self._generate_synthetic_data(synthetic_data_size)

        # Environment parameters
        self.simulation_steps = simulation_steps
        self.current_step = 0
        self.random_disruptions = random_disruptions
        self.render_mode = render_mode
        self.lead_time_weight = lead_time_weight

        # Track metrics history for visualization
        self.metrics_history = {
            "revenue": [],
            "costs": [],
            "lead_time": [],
            "defect_rate": [],
            "total_reward": [],
            "predicted_values": [],  # Track prediction accuracy
            "actual_values": [],     # Track actual values
        }

        # Get unique categorical values
        self.suppliers: Any = self.base_data["Supplier name"].unique().tolist()
        self.transport_modes: Any = (
            self.base_data["Transportation modes"].unique().tolist()
        )
        self.routes: Any = self.base_data["Routes"].unique().tolist()

        # Define observation space
        # We'll use Box for numerical features and Discrete for categorical
        self.num_features = 9  # numerical features
        self.num_suppliers = len(self.suppliers)
        self.num_transport_modes = len(self.transport_modes)
        self.num_routes = len(self.routes)

        # Total observation space dimension
        obs_dim = self.num_features + self.num_suppliers + self.num_transport_modes + self.num_routes

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Define action space
        # [supplier_idx, order_quantity, transport_mode_idx, route_idx, production_volume_adjustment]
        self.action_space = spaces.MultiDiscrete(
            [
                self.num_suppliers,  # Supplier selection
                10,  # Order quantity (scaled 1-10)
                self.num_transport_modes,  # Transportation mode
                self.num_routes,  # Route selection
                5,  # Production volume adjustment (-2, -1, 0, 1, 2)
            ]
        )

        # Initialize state
        self.state = None
        self.current_supplier_idx = 0
        self.current_transport_idx = 0
        self.current_route_idx = 0

        # Setup for rendering
        self.fig = None
        self.ax = None
        
        # Setup for value prediction models
        self.prediction_models = self._initialize_prediction_models()
        self.prediction_features = [
            "Price", 
            "Stock levels", 
            "Lead time",
            "Production volumes", 
            "Manufacturing lead time", 
            "Manufacturing costs",
            "Defect rates", 
            "Shipping costs", 
            "Costs"
        ]
        self.train_prediction_models()

    def _load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the CSV data, removing irrelevant columns."""
        data = pd.read_csv(csv_path)

        # Select relevant columns
        relevant_columns = [
            "Price",
            "Stock levels",
            "Lead time",
            "Production volumes",
            "Manufacturing lead time",
            "Manufacturing costs",
            "Defect rates",
            "Shipping costs",
            "Costs",
            "Supplier name",
            "Transportation modes",
            "Routes",
            "Revenue generated",
        ]

        # Filter only relevant columns that exist in the data
        relevant_columns = [col for col in relevant_columns if col in data.columns]
        data = data[relevant_columns]

        # Handle missing values if any
        data = data.fillna(data.mean(numeric_only=True))

        return data

    def _generate_synthetic_data(self, size: int) -> pd.DataFrame:
        """Generate synthetic data based on the original dataset distributions."""
        if size <= 0:
            return self.base_data.copy()

        synthetic_data = []

        # Get distributions of numerical features
        num_cols = [
            "Price",
            "Stock levels",
            "Lead time",
            "Production volumes",
            "Manufacturing lead time",
            "Manufacturing costs",
            "Defect rates",
            "Shipping costs",
            "Costs",
            "Revenue generated",
        ]

        # Get existing numerical columns
        num_cols = [col for col in num_cols if col in self.base_data.columns]

        # Calculate means and standard deviations
        means = self.base_data[num_cols].mean()
        stds = self.base_data[num_cols].std()

        # Get categorical columns and their values
        cat_cols = ["Supplier name", "Transportation modes", "Routes"]
        cat_cols = [col for col in cat_cols if col in self.base_data.columns]

        cat_values = {}
        for col in cat_cols:
            cat_values[col] = self.base_data[col].unique().tolist()

        # Generate synthetic samples
        for _ in range(size):
            sample = {}

            # Generate numerical features
            for col in num_cols:
                # Add some random variation but keep within reasonable bounds
                mean_val = means[col]
                std_val = stds[col]
                min_val = max(0, self.base_data[col].min())
                max_val = self.base_data[col].max() * 1.2  # Allow some growth

                # Generate value with normal distribution
                val = np.random.normal(mean_val, std_val)

                # Ensure it's within bounds
                val = max(min_val, min(max_val, val))

                sample[col] = val

            # Generate categorical features
            for col in cat_cols:
                sample[col] = random.choice(cat_values[col])

            synthetic_data.append(sample)

        # Convert to DataFrame
        synth_df = pd.DataFrame(synthetic_data)

        # Combine with original data
        combined_data = pd.concat([self.base_data, synth_df], ignore_index=True)

        return combined_data

    def _get_state_representation(self) -> np.ndarray:
        """Convert current state to the format expected by the observation space."""
        # Extract current state values
        numerical_features = [
            self.current_state["Price"],
            self.current_state["Stock levels"],
            self.current_state["Lead time"],
            self.current_state["Production volumes"],
            self.current_state["Manufacturing lead time"],
            self.current_state["Manufacturing costs"],
            self.current_state["Defect rates"],
            self.current_state["Shipping costs"],
            self.current_state["Costs"],
        ]

        # One-hot encode supplier
        supplier_one_hot = [0] * len(self.suppliers)
        supplier_one_hot[self.current_supplier_idx] = 1

        # One-hot encode transport mode
        transport_one_hot = [0] * len(self.transport_modes)
        transport_one_hot[self.current_transport_idx] = 1

        # One-hot encode route
        route_one_hot = [0] * len(self.routes)
        route_one_hot[self.current_route_idx] = 1

        # Combine all features
        state_representation = (
            numerical_features + supplier_one_hot + transport_one_hot + route_one_hot
        )

        return np.array(state_representation, dtype=np.float32)

    def _get_input_features_for_prediction(self) -> np.ndarray:
        """Extract features needed for supply chain value prediction."""
        # Extract categorical features as indices
        categorical_features = [
            self.current_supplier_idx / self.num_suppliers,
            self.current_transport_idx / self.num_transport_modes,
            self.current_route_idx / self.num_routes
        ]
        
        # Get action-related features
        # These would typically come from the most recent action
        # For simplicity, we'll use some defaults if not available
        if hasattr(self, 'last_action'):
            order_quantity = self.last_action[1] / 10.0  # Normalize to 0-1
            prod_vol_adj = (self.last_action[4] / 4.0) + 0.5  # Normalize -2 to 2 as 0 to 1
        else:
            order_quantity = 0.5
            prod_vol_adj = 0.5
            
        input_features = categorical_features + [order_quantity, prod_vol_adj]
        return np.array(input_features).reshape(1, -1)

    def _initialize_prediction_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for predicting supply chain values."""
        models = {}
        for feature in [
            "Price", "Stock levels", "Lead time", "Production volumes",
            "Manufacturing lead time", "Manufacturing costs", "Defect rates",
            "Shipping costs", "Costs"
        ]:
            models[feature] = RandomForestRegressor(n_estimators=50, random_state=42)
        return models
    
    def train_prediction_models(self):
        """Train the prediction models on the available data."""
        # Prepare training data
        X_train = []
        y_train = {feature: [] for feature in self.prediction_features}
        
        # For each data point in our dataset
        for _, row in self.full_data.iterrows():
            # Get supplier, transport, and route indices
            try:
                supplier_idx = self.suppliers.index(row["Supplier name"])
                transport_idx = self.transport_modes.index(row["Transportation modes"])
                route_idx = self.routes.index(row["Routes"])
                
                # Normalize indices to 0-1 range
                supplier_norm = supplier_idx / self.num_suppliers
                transport_norm = transport_idx / self.num_transport_modes
                route_norm = route_idx / self.num_routes
                
                # Create input feature vector (simplified for demonstration)
                # In a real system, you'd include more features like order quantities, etc.
                features = [supplier_norm, transport_norm, route_norm, 0.5, 0.5]  # Last two are placeholders
                X_train.append(features)
                
                # Get target values for each prediction model
                for feature in self.prediction_features:
                    if feature in row:
                        y_train[feature].append(row[feature])
                    else:
                        y_train[feature].append(0)  # Fallback if data is missing
            except (ValueError, KeyError):
                # Skip this row if there's an issue with the indices
                continue
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        
        # Train each model if we have enough data
        if len(X_train) > 0:
            for feature in self.prediction_features:
                y = np.array(y_train[feature])
                self.prediction_models[feature].fit(X_train, y)
                
    def predict_supply_chain_values(self) -> Dict[str, float]:
        """Predict the supply chain values based on current state and actions."""
        input_features = self._get_input_features_for_prediction()
        
        predictions = {}
        for feature in self.prediction_features:
            predicted_value = self.prediction_models[feature].predict(input_features)[0]
            # Ensure predictions are reasonable (no negative values)
            predictions[feature] = max(0, predicted_value)
            
        return predictions

    def _apply_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Apply the selected action and get the new state."""
        # Store the action for later use in predictions
        self.last_action = action
        
        supplier_idx, order_qty_scaled, transport_idx, route_idx, prod_vol_adj = action

        # Convert scaled order quantity to actual value (1-10 to actual range)
        min_order = 10
        max_order = 100
        order_quantity = min_order + (max_order - min_order) * (order_qty_scaled / 10.0)

        # Convert production volume adjustment (-2 to 2) to percentage change
        prod_adj_percentages = [-0.2, -0.1, 0, 0.1, 0.2]  # -20% to +20%
        prod_vol_change = prod_adj_percentages[prod_vol_adj]

        # Update current indices
        self.current_supplier_idx = supplier_idx
        self.current_transport_idx = transport_idx
        self.current_route_idx = route_idx

        # Get current supplier, transport mode and route
        current_supplier = self.suppliers[supplier_idx]
        current_transport = self.transport_modes[transport_idx]
        current_route = self.routes[route_idx]

        # Filter data based on selections
        filtered_data = self.full_data[
            (self.full_data["Supplier name"] == current_supplier)
            & (self.full_data["Transportation modes"] == current_transport)
            & (self.full_data["Routes"] == current_route)
        ]

        # If no exact match, get the closest available option
        if filtered_data.empty:
            # Get data for just the supplier
            supplier_data = self.full_data[
                self.full_data["Supplier name"] == current_supplier
            ]

            if supplier_data.empty:
                # Fallback to a random data point
                filtered_data = self.full_data.sample(1)
            else:
                filtered_data = supplier_data.sample(1)

        # Start with a random state from the filtered data
        new_state = filtered_data.sample(1).iloc[0].to_dict()

        # Apply production volume adjustment
        current_prod_vol = new_state["Production volumes"]
        new_prod_vol = current_prod_vol * (1 + prod_vol_change)
        new_state["Production volumes"] = new_prod_vol

        # Apply order quantity effect on stock levels and lead time
        # Higher order quantity can reduce stock levels and increase lead time
        stock_effect = -0.1 * (order_quantity / max_order)  # -10% at max order
        lead_time_effect = 0.15 * (order_quantity / max_order)  # +15% at max order

        new_state["Stock levels"] *= 1 + stock_effect
        new_state["Lead time"] *= 1 + lead_time_effect

        # Apply random disruptions if enabled
        if (
            self.random_disruptions and random.random() < 0.1
        ):  # 10% chance of disruption
            disruption_type = random.choice(["supply", "transport", "quality"])

            if disruption_type == "supply":
                # Supply disruption affects lead time and stock levels
                new_state["Lead time"] *= random.uniform(1.1, 1.5)  # 10-50% increase
                new_state["Stock levels"] *= random.uniform(0.6, 0.9)  # 10-40% decrease

            elif disruption_type == "transport":
                # Transport disruption affects shipping costs and times
                new_state["Shipping costs"] *= random.uniform(
                    1.1, 1.3
                )  # 10-30% increase

            elif disruption_type == "quality":
                # Quality issues affect defect rates
                new_state["Defect rates"] *= random.uniform(
                    1.1, 2.0
                )  # 10-100% increase

        # Calculate revenue based on production volume, price, and defect rate
        effective_production = new_state["Production volumes"] * (
            1 - new_state["Defect rates"] / 100
        )
        new_state["Revenue generated"] = effective_production * new_state["Price"]

        # Calculate total costs
        new_state["Costs"] = (
            new_state["Manufacturing costs"] * new_state["Production volumes"]
            + new_state["Shipping costs"] * effective_production
        )

        return new_state

    def _calculate_reward(self, state: Dict[str, Any]) -> float:
        """Calculate reward based on key performance indicators with emphasis on lead time."""
        # Extract metrics
        revenue = state["Revenue generated"]
        total_cost = state["Costs"]
        lead_time = state["Lead time"]
        manufacturing_lead_time = state["Manufacturing lead time"]
        defect_rate = state["Defect rates"]

        # Calculate profit
        profit = revenue - total_cost

        # Normalize metrics for reward calculation
        max_revenue = self.full_data["Revenue generated"].max()
        max_cost = self.full_data["Costs"].max()
        max_lead_time = self.full_data["Lead time"].max()
        max_manufacturing_lead_time = self.full_data["Manufacturing lead time"].max()
        max_defect_rate = self.full_data["Defect rates"].max()

        norm_profit = profit / max_revenue  # Normalize to approximately -1 to 1 range
        norm_lead_time = 1 - (lead_time / max_lead_time)  # Lower is better
        norm_manufacturing_lead_time = 1 - (manufacturing_lead_time / max_manufacturing_lead_time)  # Lower is better
        norm_defect_rate = 1 - (defect_rate / max_defect_rate)  # Lower is better

        # Weight the components with emphasis on lead times
        # lead_time_weight is now a parameter that can be adjusted during initialization
        remaining_weight = 1.0 - self.lead_time_weight
        profit_weight = 0.5 * remaining_weight
        defect_rate_weight = 0.5 * remaining_weight
        
        # Divide lead time weight between supplier lead time and manufacturing lead time
        supplier_lead_time_weight = 0.6 * self.lead_time_weight
        manufacturing_lead_time_weight = 0.4 * self.lead_time_weight

        reward = (
            profit_weight * norm_profit + 
            supplier_lead_time_weight * norm_lead_time + 
            manufacturing_lead_time_weight * norm_manufacturing_lead_time + 
            defect_rate_weight * norm_defect_rate
        )

        return reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Reset metrics history
        self.metrics_history = {
            "revenue": [],
            "costs": [],
            "lead_time": [],
            "defect_rate": [],
            "total_reward": [],
            "predicted_values": [],
            "actual_values": [],
        }

        # Choose a random initial state from the dataset
        self.current_state = self.full_data.sample(1).iloc[0].to_dict()

        # Initialize supplier, transport, and route indices
        supplier = self.current_state["Supplier name"]
        transport = self.current_state["Transportation modes"]
        route = self.current_state["Routes"]

        self.current_supplier_idx = self.suppliers.index(supplier)
        self.current_transport_idx = self.transport_modes.index(transport)
        self.current_route_idx = self.routes.index(route)

        # Get state representation
        state = self._get_state_representation()

        # Clear the existing figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        if self.render_mode == "human":
            self._render_frame()

        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment based on the action."""
        # Predict supply chain values before applying action
        predicted_values = self.predict_supply_chain_values()

        # Apply action to get new state
        self.current_state = self._apply_action(action)

        # Calculate reward
        reward = self._calculate_reward(self.current_state)

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.simulation_steps
        truncated = False

        # Compare predicted values with actual values
        prediction_errors = {}
        for feature in self.prediction_features:
            if feature in self.current_state:
                actual = self.current_state[feature]
                predicted = predicted_values[feature]
                error = abs(actual - predicted) / (actual + 1e-10)  # Relative error
                prediction_errors[feature] = error

        # Update metrics history
        self.metrics_history["revenue"].append(self.current_state["Revenue generated"])
        self.metrics_history["costs"].append(self.current_state["Costs"])
        self.metrics_history["lead_time"].append(self.current_state["Lead time"])
        self.metrics_history["defect_rate"].append(self.current_state["Defect rates"])
        self.metrics_history["total_reward"].append(reward)
        self.metrics_history["predicted_values"].append(predicted_values)
        self.metrics_history["actual_values"].append({k: self.current_state.get(k, 0) for k in self.prediction_features})

        # Get observation
        observation = self._get_state_representation()

        # Additional info
        info = {
            "profit": self.current_state["Revenue generated"] - self.current_state["Costs"],
            "supplier": self.suppliers[self.current_supplier_idx],
            "transport_mode": self.transport_modes[self.current_transport_idx],
            "route": self.routes[self.current_route_idx],
            "metrics": {
                "revenue": self.current_state["Revenue generated"],
                "costs": self.current_state["Costs"],
                "lead_time": self.current_state["Lead time"],
                "defect_rate": self.current_state["Defect rates"],
                "production_volume": self.current_state["Production volumes"],
            },
            "predictions": predicted_values,
            "prediction_errors": prediction_errors
        }

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        """Render the current state of the environment."""
        # Initialize the figure and axes if they don't exist
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(3, 2, figsize=(14, 10))
            plt.tight_layout(pad=3.0)
            self.fig.suptitle("Meridian Engine Metrics", fontsize=16)

        # Clear previous plots
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()

        # Plot metrics over time
        steps = list(range(1, len(self.metrics_history["revenue"]) + 1))

        if steps:  # Ensure we have data to plot
            # Revenue and Cost
            self.ax[0, 0].plot(
                steps, self.metrics_history["revenue"], "g-", label="Revenue"
            )
            self.ax[0, 0].plot(
                steps, self.metrics_history["costs"], "r-", label="Costs"
            )
            self.ax[0, 0].set_title("Revenue & Costs")
            self.ax[0, 0].set_xlabel("Step")
            self.ax[0, 0].set_ylabel("Amount")
            self.ax[0, 0].legend()

            # Lead Time
            self.ax[0, 1].plot(steps, self.metrics_history["lead_time"], "b-")
            self.ax[0, 1].set_title("Lead Time")
            self.ax[0, 1].set_xlabel("Step")
            self.ax[0, 1].set_ylabel("Days")

            # Defect Rate
            self.ax[1, 0].plot(steps, self.metrics_history["defect_rate"], "m-")
            self.ax[1, 0].set_title("Defect Rate")
            self.ax[1, 0].set_xlabel("Step")
            self.ax[1, 0].set_ylabel("Percentage")

            # Reward
            self.ax[1, 1].plot(steps, self.metrics_history["total_reward"], "k-")
            self.ax[1, 1].set_title("Total Reward")
            self.ax[1, 1].set_xlabel("Step")
            self.ax[1, 1].set_ylabel("Reward")
            
            # Prediction accuracy
            if len(self.metrics_history["predicted_values"]) > 0:
                # Calculate prediction errors for selected metrics
                lead_time_errors = []
                cost_errors = []
                
                for i in range(len(self.metrics_history["predicted_values"])):
                    pred = self.metrics_history["predicted_values"][i]
                    actual = self.metrics_history["actual_values"][i]
                    
                    if "Lead time" in pred and "Lead time" in actual:
                        error = abs(pred["Lead time"] - actual["Lead time"]) / (actual["Lead time"] + 1e-10)
                        lead_time_errors.append(min(error, 1.0))  # Cap at 100% error for visualization
                    
                    if "Costs" in pred and "Costs" in actual:
                        error = abs(pred["Costs"] - actual["Costs"]) / (actual["Costs"] + 1e-10)
                        cost_errors.append(min(error, 1.0))  # Cap at 100% error for visualization
                
                # Plot prediction errors
                if lead_time_errors:
                    self.ax[2, 0].plot(steps[-len(lead_time_errors):], lead_time_errors, 'r-', label="Lead Time")
                if cost_errors:
                    self.ax[2, 0].plot(steps[-len(cost_errors):], cost_errors, 'b-', label="Costs")
                
                self.ax[2, 0].set_title("Prediction Error (%)")
                self.ax[2, 0].set_xlabel("Step")
                self.ax[2, 0].set_ylabel("Relative Error")
                self.ax[2, 0].legend()
                self.ax[2, 0].set_ylim(0, 1.0)
                
                # Feature importance
                if hasattr(self.prediction_models["Lead time"], 'feature_importances_'):
                    importances = self.prediction_models["Lead time"].feature_importances_
                    feature_names = ["Supplier", "Transport", "Route", "Order Qty", "Prod Vol"]
                    
                    # Sort features by importance
                    indices = np.argsort(importances)
                    self.ax[2, 1].barh(range(len(indices)), importances[indices], color='b')
                    self.ax[2, 1].set_yticks(range(len(indices)))
                    self.ax[2, 1].set_yticklabels([feature_names[i] for i in indices])
                    self.ax[2, 1].set_title("Lead Time Prediction\nFeature Importance")

            # Current supplier and transport info
            supplier = self.suppliers[self.current_supplier_idx]
            transport = self.transport_modes[self.current_transport_idx]
            route = self.routes[self.current_route_idx]

            plt.figtext(
                0.5,
                0.01,
                f"Step: {self.current_step}/{self.simulation_steps} | "
                f"Supplier: {supplier} | Transport: {transport} | Route: {route}",
                ha="center",
                fontsize=10,
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
            )

        plt.draw()
        plt.pause(0.1)

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        """Close the environment and clean up resources."""
        try:
            if self.fig is not None:
                plt.close(self.fig)
                plt.ioff()
                self.fig = None
                self.ax = None
        except Exception as e:
            self.logger.error(f"Error closing matplotlib figure: {e}")
