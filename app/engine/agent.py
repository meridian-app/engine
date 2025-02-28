from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

class SupplyChainAgent:
    """
    Q-Learning agent with epsilon-greedy exploration for supply chain optimization
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        discretization_bins: int = 10,
    ) -> None:
        """
        Initialize the Q-Learning agent.

        Args:
            env: The Gymnasium environment
            learning_rate: Alpha - learning rate for Q-value updates
            discount_factor: Gamma - discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays per episode
            discretization_bins: Number of bins to discretize continuous state space
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Get action space details
        self.action_space = env.action_space

        # Set up discretization for continuous state space
        self.discretization_bins = discretization_bins

        # Dictionary-based Q-table for sparse representation
        # Using Any for the type to avoid specific numpy type constraints
        self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()))

        # Track training metrics
        self.rewards_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.profit_history: List[float] = []

    def get_action_space_size(self) -> int:
        """Calculate total number of possible actions"""
        # For MultiDiscrete action space, multiply all dimensions
        return int(np.prod(self.action_space.nvec))

    def index_to_action(self, action_idx: int) -> np.ndarray:
        """Convert a flat action index to a MultiDiscrete action vector"""
        # Ensure action_idx is an int
        action_idx = int(action_idx)
        action = []
        remaining = action_idx

        # Convert flat index to multi-index
        for dim_size in reversed(self.action_space.nvec):
            action.insert(0, remaining % dim_size)
            remaining //= dim_size

        return np.array(action)

    def action_to_index(self, action: np.ndarray) -> int:
        """Convert a MultiDiscrete action vector to a flat index"""
        idx = 0
        cumulative_dim = 1

        # Convert multi-index to flat index
        for i in range(len(self.action_space.nvec) - 1, -1, -1):
            idx += action[i] * cumulative_dim
            cumulative_dim *= self.action_space.nvec[i]

        return int(idx)  # Ensure the result is an int

    def discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state space to make it suitable for Q-table lookup.

        For categorical features (one-hot encoded), we'll preserve the values.
        For continuous features, we'll bin them into discrete values.
        """
        # Determine which features are continuous (first 9 features in our environment)
        num_continuous = 9

        # Get min and max values for normalization
        # Note: In a real implementation, you might want to set these based on domain knowledge
        # or by tracking min/max across episodes
        feature_min = np.array([-100.0] * num_continuous)  # Arbitrary low values
        feature_max = np.array([1000.0] * num_continuous)  # Arbitrary high values

        # Calculate bin edges for each continuous feature
        discretized_state = []

        # Discretize continuous features
        for i in range(num_continuous):
            # Clip value to min/max range
            value = np.clip(state[i], feature_min[i], feature_max[i])

            # Normalize to 0-1 range
            normalized = (value - feature_min[i]) / (feature_max[i] - feature_min[i])

            # Discretize to bin
            bin_idx = min(
                int(normalized * self.discretization_bins), self.discretization_bins - 1
            )
            discretized_state.append(bin_idx)

        # Keep categorical features (one-hot encoded) as is - just get indices of non-zero values
        for i in range(num_continuous, len(state)):
            discretized_state.append(1 if state[i] > 0.5 else 0)

        return tuple(discretized_state)

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: The current state

        Returns:
            The selected action as a numpy array
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # Exploitation: best known action
        discretized_state = self.discretize_state(state)
        action_idx = int(np.argmax(self.q_table[discretized_state]))  # Convert to int explicitly
        return self.index_to_action(action_idx)

    def update_q_table(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update Q-table using the Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert state and action to indices for Q-table lookup
        state_key = self.discretize_state(state)
        action_idx = self.action_to_index(action)

        # Get current Q-value
        current_q = self.q_table[state_key][action_idx]

        # Calculate the new Q-value
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            next_state_key = self.discretize_state(next_state)
            best_next_action = int(np.argmax(self.q_table[next_state_key]))  # Convert to int
            target_q = (
                reward
                + self.discount_factor * self.q_table[next_state_key][best_next_action]
            )

        # Update Q-value
        self.q_table[state_key][action_idx] += self.learning_rate * (
            target_q - current_q
        )

    def train(self, num_episodes: int = 500, max_steps: int = 30) -> Dict:
        """
        Train the agent for a specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode

        Returns:
            Dict containing training history
        """
        self.rewards_history = []
        self.epsilon_history = []
        self.profit_history = []

        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            # Reset environment for new episode
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0

            for step in range(max_steps):
                # Choose and perform action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Update Q-table
                done = terminated or truncated
                self.update_q_table(state, action, reward, next_state, done)

                # Track metrics
                episode_reward += reward
                episode_profit += info.get("profit", 0)

                # Update state
                state = next_state

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Record metrics
            self.rewards_history.append(episode_reward)
            self.epsilon_history.append(self.epsilon)
            self.profit_history.append(episode_profit)

        return {
            "rewards": self.rewards_history,
            "epsilon": self.epsilon_history,
            "profit": self.profit_history,
        }

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained agent without exploration (epsilon = 0).

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dict containing evaluation metrics
        """
        saved_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation

        evaluation_rewards = []
        evaluation_profits = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            done = False

            while not done:
                # Choose action based on learned policy
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_profit += info.get("profit", 0)
                done = terminated or truncated
                state = next_state

            evaluation_rewards.append(episode_reward)
            evaluation_profits.append(episode_profit)

        # Restore epsilon
        self.epsilon = saved_epsilon

        return {
            "mean_reward": np.mean(evaluation_rewards),
            "std_reward": np.std(evaluation_rewards),
            "mean_profit": np.mean(evaluation_profits),
            "std_profit": np.std(evaluation_profits),
        }

    def plot_training_results(self) -> None:
        """Plot training metrics"""
        plt.figure(figsize=(15, 10))

        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.rewards_history)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot smoothed rewards
        plt.subplot(2, 2, 2)
        window_size = min(50, len(self.rewards_history) // 10 + 1)
        smoothed_rewards = (
            pd.Series(self.rewards_history).rolling(window=window_size).mean()
        )
        plt.plot(smoothed_rewards)
        plt.title(f"Smoothed Rewards (Window Size: {window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        # Plot profit
        plt.subplot(2, 2, 4)
        plt.plot(self.profit_history)
        plt.title("Episode Profit")
        plt.xlabel("Episode")
        plt.ylabel("Profit")

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the Q-table to a file"""
        # Convert defaultdict to dict for saving
        q_dict = dict(self.q_table)

        # Convert tuple keys to string
        serializable_q = {str(k): v.tolist() for k, v in q_dict.items()}

        # Save to file
        import json

        with open(filepath, "w") as f:
            json.dump(serializable_q, f)

    def load_model(self, filepath: str) -> None:
        """Load Q-table from a file"""
        import json
        import ast

        with open(filepath, "r") as f:
            serialized_q = json.load(f)

        # Convert string keys back to tuples and ensure proper type handling
        for k, v in serialized_q.items():
            # Use ast.literal_eval to safely convert string to tuple
            try:
                # First attempt: try to parse as a tuple directly
                key_tuple = ast.literal_eval(k)
                if not isinstance(key_tuple, tuple):
                    # If not a tuple, try alternative parsing
                    key_tuple = tuple(map(int, k.strip("()").split(", ")))
            except (ValueError, SyntaxError):
                # Fallback: manual parsing
                key_tuple = tuple(map(int, k.strip("()").split(", ")))
                
            # Create a fresh numpy array to avoid type issues
            self.q_table[key_tuple] = np.array(v, dtype=np.float64) # type: ignore