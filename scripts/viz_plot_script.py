import numpy as np
import matplotlib.pyplot as plt

# Mapping from agent names to their labels for the legend in the plot
plt_legend_dict = {"expected_sarsa_agent": "Expected SARSA with neural network"}

# Directory where the results are stored
path_dict = {"expected_sarsa_agent": "data/results/"}

# Label for the y-axis in the plot
y_axis_label = "Reward"

def smooth(data, k):
    """
    Smooths the data using a sliding window of size k.

    Parameters:
    - data: numpy array of shape (num_runs, num_episodes)
    - k: window size for smoothing

    Returns:
    - smoothed_data: numpy array of shape (num_runs, num_episodes) after smoothing
    """
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    # Initialize an array to hold the smoothed data
    smoothed_data = np.zeros((num_runs, num_episodes))

    # Apply smoothing
    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i+1], axis=1)   
        else:
            smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis=1)    

    return smoothed_data

def plot_result(data_name_array):
    """
    Plots the learning curve for the agents specified in data_name_array.

    Parameters:
    - data_name_array: list of strings, names of the agents to plot
    """
    plt_agent_sweeps = []
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8,6))

    # Plot data for each agent
    for data_name in data_name_array:
        # Load the sum of rewards data
        filename = 'sum_reward_{}'.format(data_name).replace('.', '')
        sum_reward_data = np.load('{}/{}.npy'.format(path_dict[data_name], filename))

        # Smooth the data
        smoothed_sum_reward = smooth(data=sum_reward_data, k=100)
        
        # Compute the mean of the smoothed data across runs
        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis=0)

        # Plot the learning curve
        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph, = ax.plot(plot_x_range, mean_smoothed_sum_reward, label=plt_legend_dict[data_name])
        plt_agent_sweeps.append(graph)
    
    # Configure the plot
    ax.legend(handles=plt_agent_sweeps, fontsize=13)
    ax.set_title("Learning Curve", fontsize=15)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel(y_axis_label, rotation=90, labelpad=40, fontsize=14)
    ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.savefig("data/results/learning_curve.png")
    plt.clf()

# Call the function to plot the results for the specified agent(s)
plot_result(["expected_sarsa_agent"])