from src.utils.optimizer import Adam
from src.models.replay_buffer import ReplayBuffer
from src.models.network import ActionValueNetwork
from src.agents.base_agent import BaseAgent
import numpy as np
from copy import deepcopy
from src.utils.softmax import mathematics
import json

# Utility for mathematical operations like Softmax and TD-error calculations
soft = mathematics()

class Agent(BaseAgent):
    """
    Deep Expected SARSA Agent implementation for the Lunar Lander environment.
    
    This agent utilizes a multi-layer perceptron (ActionValueNetwork) to approximate 
    Q-values and employs Expected SARSA for off-policy control. Key stability 
    features include a Replay Buffer for experience replay and a Target Network 
    with Soft Updates (Polyak Averaging) to prevent catastrophic forgetting.
    """

    def __init__(self):
        """Initialize the agent with a default name."""
        super().__init__()
        self.name = "expected_sarsa_agent"
        
    def agent_init(self, agent_config):
        """
        Setup for the agent called when the experiment first starts.

        Args:
            agent_config (dict): Configuration parameters including:
                network_config (dict): Architecture and seed for the NN.
                replay_buffer_size (int): Max capacity of the experience replay.
                minibatch_sz (int): Number of experiences per gradient update.
                num_replay_updates_per_step (int): Training iterations per environment step.
                gamma (float): Discount factor for future rewards.
                tau (float): Temperature for Softmax policy.
                tau_soft (float): Interpolation factor for soft target updates.
        """
        # Initialize experience replay memory to break temporal correlations
        self.replay_buffer = ReplayBuffer(
            agent_config['replay_buffer_size'], 
            agent_config['minibatch_sz'], 
            agent_config["network_config"]["seed"]
        )

        # Main network used for active learning and action selection
        self.network = ActionValueNetwork(agent_config['network_config'])
        
        # Optimizer (Adam) to manage weight updates
        self.optimizer = Adam(self.network.layer_sizes)
        
        # Hyperparameters
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        self.tau_soft = agent_config.get('tau_soft', 0.005)
        
        self.rand_generator = np.random.RandomState(agent_config["network_config"]["seed"])
        
        # Target Network: A slowly tracking copy of the main network to stabilize TD-targets
        self.target_network = deepcopy(self.network)
        
        # Trackers for synchronization and performance metrics
        self.total_steps = 0
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0

    def policy(self, state):
        """
        Selects an action based on a Softmax probability distribution.

        Args:
            state (Numpy array): The current state observation.
        Returns:
            int: The selected action.
        """
        # Retrieve action values from the current policy network
        action_values = self.network.get_action_values(state)
        
        # Convert Q-values to probabilities via Softmax (controlled by temperature tau)
        probs_batch = soft.softmax(action_values, self.tau)
        
        # Sample action from the computed distribution (ensure 1D for choice)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.flatten())
        return action

    def synchronize_target_network(self):
        """
        Performs a Soft Update (Polyak Averaging) on the Target Network weights.
        
        Instead of hard-copying weights, the target network slowly tracks the 
        active network using: theta_target = tau_soft * theta_local + (1 - tau_soft) * theta_target.
        This provides a stable moving target for the TD-error calculation.
        """
        local_weights = self.network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            for key in target_weights[i]:
                # Blend the local weights into the target weights
                target_weights[i][key] = (self.tau_soft * local_weights[i][key] + 
                                        (1 - self.tau_soft) * target_weights[i][key])
        
        self.target_network.set_weights(target_weights)

    def agent_start(self, state):
        """
        Called at the beginning of an episode.

        Args:
            state (Numpy array): The first state observation from env.reset().
        Returns:
            int: The agent's first action.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        
        # Ensure state has batch dimension for consistency with the network input shape
        self.last_state = np.array([state])
        
        # Select first action using current policy
        self.last_action = self.policy(self.last_state)
        
        return self.last_action

    def agent_step(self, reward, state):
        """
        A standard step in the environment.

        Args:
            reward (float): The reward received from the previous action.
            state (Numpy array): The new state observation.
        Returns:
            int: The next action.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        self.total_steps += 1

        # Add batch dimension to current state
        state = np.array([state])

        # Choose next action (Expected SARSA is off-policy, but typically uses same policy)
        action = self.policy(state)
        
        # Store experience in Replay Buffer: (S, A, R, Terminal_Flag, S')
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state) 
        
        # Learning phase: Perform gradient updates using random samples from replay memory
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                
                # Sample a random minibatch of experiences
                experiences = self.replay_buffer.sample()
                
                # Update main network weights using Expected SARSA targets from the Target Network
                soft.optimize_network(experiences, self.discount, self.optimizer, 
                                     self.network, self.target_network, self.tau)
                
                # Gently update target network after each optimization step
                self.synchronize_target_network()
                
        # Transition tracking
        self.last_state = state
        self.last_action = action

        return action

    def agent_end(self, reward):
        """
        Final update called when an episode terminates.

        Args:
            reward (float): The final reward received upon reaching a terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        self.total_steps += 1
        
        # Terminal next-state is traditionally represented as all zeros for calculation consistency
        state = np.zeros_like(self.last_state)

        # Append terminal experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Learning phase on episode end
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                soft.optimize_network(experiences, self.discount, self.optimizer, 
                                     self.network, self.target_network, self.tau)
                self.synchronize_target_network()
        
    def agent_message(self, message):
        """
        Handles communication between the experiment runner and the agent.
        
        Args:
            message (str): The requested information or action.
        """
        if message == "get_sum_reward":
            return self.sum_rewards
        elif message == "get_weights":
            return self.network.get_weights()
        else:
            raise Exception(f"Unrecognized Message: {message}")
        
    def set_weights(self, weights):
        """
        Overrides the current network weights with provided values.

        Args:
            weights (list): Collection of dictionaries containing 'W' and 'b' for each layer.
        """
        self.network.set_weights(weights)
        self.target_network = deepcopy(self.network)
