import imageio
import gymnasium as gym
import numpy as np
from src.agents.expected_sarsa_agent import Agent
import json
import time
import matplotlib.pyplot as plt


# Load agent configuration from a JSON file
with open("configs/demo_parameters.json", "r") as f:
    agent_config = json.load(f)

timeout = agent_config["experiment_parameters"]["timeout"]

# Initialize the agent with the loaded configuration
agent = Agent()
agent.agent_init(agent_config=agent_config)

# Load the trained agent's weights
weights_path = "data/weights/weights_expected_sarsa_agent.npy"
weights = np.load(weights_path, allow_pickle=True)

# Set the pretrained agent's weights
agent.set_weights(weights)

def test_agent_and_save_frames(env_name, num_episodes=1):
    """
    Tests the agent on a given environment and saves the frames for video creation.

    Parameters:
    - env_name: Name of the environment to test the agent on.
    - num_episodes: Number of episodes to run the agent for.

    Returns:
    - frames: A list of frames captured during the agent's episode(s).
    """
    # Create the environment with specified rendering mode
    env = gym.make(env_name, render_mode='rgb_array')
    frames = []

    for episode in range(num_episodes):
        start_time = time.time()
        state, info = env.reset()
        action = agent.agent_start(state)  # Start the episode with the agent
        done = False
        while not done:
            if time.time() - start_time > timeout:
                print("Timeout reached, ending episode.")
                break
            frame = env.render()  # Capture the current frame
            frames.append(frame)
            state, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
            done = terminated or truncated
            if not done:
                action = agent.agent_step(reward, state)  # Update the agent's state
            else:
                agent.agent_end(reward)  # End the episode for the agent if needed
    env.close()  # Clean up the environment
    return frames

def create_video_from_frames(frames, output_file='agent_performance.mp4', fps=30):
    """
    Creates a video from a list of frames.

    Parameters:
    - frames: A list of frames to compile into a video.
    - output_file: The filename for the output video.
    - fps: Frames per second for the output video.
    """
    writer = imageio.get_writer(output_file, fps=fps)
    for frame in frames:
        writer.append_data(frame)  # Append each frame to the video
    writer.close()

# Execute the functions to test the agent and create a video
frames = test_agent_and_save_frames('LunarLander-v3', 1)
create_video_from_frames(frames, 'docs/videos/agent_performance.mp4', 30)