import numpy as np
from src.utils.rl_glue import RLGlue
from src.environments.lunar_lander import LunarLanderEnvironment
from src.agents.expected_sarsa_agent import Agent
from tqdm import tqdm
import os 
import shutil
import json

# Correctly locate the config file from project root
with open("configs/parameters.json", "r") as f:
    config = json.load(f)


# Actions for LunarLander-v2
# 0 = Do nothing
# 1 = Fire left orientation engine
# 2 = Fire main engine
# 3 = Fire right orientation engine
# Total of 4 outputs


def run_experiment(environment, agent):

    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((config["experiment_parameters"]["num_runs"], 
                                 config["experiment_parameters"]["num_episodes"]))

    env_info = {}

    agent_info = config

    # one agent setting
    for run in range(1, config["experiment_parameters"]["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)

        for episode in tqdm(range(1, config["experiment_parameters"]["num_episodes"]+1)):
            # run episode
            # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after
            # some number of timesteps. Here we use the default of 500.
            rl_glue.rl_episode(config["experiment_parameters"]["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward

    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('data/results'):
        os.makedirs('data/results')
    if not os.path.exists('data/weights'):
        os.makedirs('data/weights')

    #### Save model weights as np.array
    weights = rl_glue.rl_agent_message("get_weights")
    # Specify the path for saving weights
    weights_path = "data/weights/weights_{}.npy".format(save_name)
    np.save(weights_path, weights)

    #### Save sum reward np.array file
    np.save("data/results/sum_reward_{}".format(save_name), agent_sum_reward)

# Run Experiment

current_env = LunarLanderEnvironment

current_agent = Agent

# run experiment
run_experiment(current_env, current_agent)