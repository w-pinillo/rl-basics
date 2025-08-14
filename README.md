# Reinforcement Learning Basics

This project is a space to explore and learn the fundamentals of Reinforcement Learning (RL). It provides a structured environment to implement and test various RL algorithms on different environments.

## Project Structure

The project is organized as follows:

```
.
├── notebooks/         # Jupyter notebooks for experiments and analysis
├── README.md          # This file
├── main.py            # Main entry point for running experiments
├── requirements.txt   # Project dependencies
└── src/
    ├── agents/        # RL agent implementations (e.g., Q-learning, DQN)
    ├── algorithms/    # Core RL algorithm logic
    ├── environments/  # Custom environment definitions
    └── utils/         # Utility functions and helper classes
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rl-basics
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run an experiment, you can use the `main.py` script. For example:

```bash
python src/main.py --agent q-learning --environment cartpole
```

