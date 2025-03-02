# Meridian Engine

A powerful supply chain prediction and optimization engine powered by reinforcement learning.

## ⚠️ Warning: Research Project Only

> **DISCLAIMER**: This project is NOT ready for production use. DO NOT USE in real-world supply chain systems.

The Meridian Engine is an **educational exploration** of AI capabilities in supply chain environments with significant limitations:

- **In-Memory Only**: All data is stored in memory and will be lost when the server restarts
- **Single Environment**: Does not properly support multiple concurrent supply chain environments
- **Limited Testing**: Has not been thoroughly tested with real-world data volumes
- **Simplified Model**: Uses a highly constrained supply chain model for educational purposes
- **Research Focus**: Created to understand reinforcement learning applications rather than for production deployment


## Overview

Meridian Engine is an AI-driven supply chain optimization platform that uses reinforcement learning techniques to make intelligent decisions about supplier selection, order quantities, transportation modes, and production volumes. The engine continuously learns from supply chain data to improve its recommendations over time.

## Features

- **Real-time Optimization**: Dynamically optimize supply chain decisions as new data becomes available
- **Reinforcement Learning**: Q-learning agent that continuously improves decision-making
- **Monte Carlo Simulation**: Pre-train environment for more accurate predictions
- **WebSocket API**: Real-time communication with client applications
- **Action Explanations**: Human-readable explanations for recommended actions
- **Visualization**: Performance metrics and prediction accuracy visualization

## Technical Architecture

The Meridian Engine consists of several key components:

1. **Supply Chain Environment**: A Gymnasium-based simulation environment that models the dynamics of a supply chain network
2. **Q-Learning Agent**: An intelligent agent that learns optimal policies through experience
3. **FastAPI Server**: A web server that exposes the optimization engine via WebSockets
4. **Prediction Models**: Machine learning models that predict key supply chain metrics

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/meridian-app/engine.git meridian-engine
    cd meridian-engine
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:

    ```bash
    fastapi dev
    ```

## Contributing

We welcome contributions to the Meridian Engine project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
