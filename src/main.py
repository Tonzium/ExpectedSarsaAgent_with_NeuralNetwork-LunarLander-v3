import numpy as np
from src.utils.rl_glue import RLGlue
from src.environments.lunar_lander import LunarLanderEnvironment
from src.agents.expected_sarsa_agent import Agent
from tqdm import tqdm
import os 
import json

"""
Lunar Lander Experiment Runner

This script initializes the environment and agent, configures their hyperparameters 
via 'parameters.json', and executes a multi-run, multi-episode experiment using the 
RLGlue framework. It saves final reward metrics and trained model weights 
to the 'data/' directory.
"""

# Load experimental hyperparameters from the configurations directory
try:
    with open("configs/parameters.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    raise Exception("Configuration file 'configs/parameters.json' not found. Please ensure it exists.")

def run_experiment(environment_class, agent_class):
    """
    Executes a Reinforcement Learning experiment by gluing an environment and agent.

    This function performs the following steps:
    1. Initializes the RLGlue interface.
    2. Iterates through multiple independent runs for statistical significance.
    3. Runs a sequence of episodes for each run using a specified timeout.
    4. Saves the training results (sum of rewards) and the final trained weights.

    Args:
        environment_class (type): The class of the environment to be instantiated.
        agent_class (type): The class of the RL agent to be instantiated.
    """
    
    # RLGlue simplifies the interaction between the agent and the environment
    rl_glue = RLGlue(environment_class, agent_class)

    # Pre-allocate array to track total rewards per episode for each run
    # Shape: (num_runs, num_episodes)
    agent_sum_reward = np.zeros((config["experiment_parameters"]["num_runs"], 
                                 config["experiment_parameters"]["num_episodes"]))

    env_info = {}
    agent_info = config

    print(f"Starting Experiment: {config['experiment_parameters']['num_runs']} runs, "
          f"{config['experiment_parameters']['num_episodes']} episodes per run.")

    # Execute multiple independent experimental runs
    for run in range(1, config["experiment_parameters"]["num_runs"] + 1):
        # Set deterministic seeds for reproducibility across runs
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        # Initialize environment and agent for the current run
        rl_glue.rl_init(agent_info, env_info)

        # Iterate through episodes
        for episode in tqdm(range(1, config["experiment_parameters"]["num_episodes"] + 1), 
                           desc=f"Run {run}"):
            
            # Run a single episode until terminal state or timeout (default 500 steps)
            rl_glue.rl_episode(config["experiment_parameters"]["timeout"])
            
            # Extract total accumulated reward for the current episode from the agent
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward

    # Ensure output directories exist
    os.makedirs('data/results', exist_ok=True)
    os.makedirs('data/weights', exist_ok=True)

    # Save final trained weights as a NumPy array for future testing/visualization
    weights = rl_glue.rl_agent_message("get_weights")
    weights_path = f"data/weights/weights_{rl_glue.agent.name}.npy"
    np.save(weights_path, weights)
    print(f"Weights saved successfully to: {weights_path}")

    # Save cumulative reward metrics for performance analysis
    results_path = f"data/results/sum_reward_{rl_glue.agent.name}.npy"
    np.save(results_path, agent_sum_reward)
    print(f"Results saved successfully to: {results_path}")

if __name__ == "__main__":
    # Define the core components of our Reinforcement Learning system
    current_env = LunarLanderEnvironment
    current_agent = Agent

    # Execute the experiment
    run_experiment(current_env, current_agent)
