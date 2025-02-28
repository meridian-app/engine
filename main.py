from app.environments.engine import SupplyChainEngine
from app.utils.engine import monte_carlo_optimization


def main():
    # Create environment
    env = SupplyChainEngine(render_mode="human")

    # Simple random agent for demonstration
    episodes = 20

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0

        for step in range(env.simulation_steps):
            # Sample random action
            action = env.action_space.sample()

            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")

    # Example of using Monte Carlo optimization
    best_action, expected_reward = monte_carlo_optimization(env)
    print(f"Best action found: {best_action}, Expected reward: {expected_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
