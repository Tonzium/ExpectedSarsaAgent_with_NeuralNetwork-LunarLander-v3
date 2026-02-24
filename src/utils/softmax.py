import numpy as np

class mathematics:
    def __init__(self):
        pass

    def softmax(self, action_values, tau=1.0):
        """
        Args:
            action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                        The action-values computed by an action-value network.              
            tau (float): The temperature parameter scalar.
        Returns:
            A 2D array of shape (batch_size, num_actions). Where each row is a probability distribution over
            the actions representing the policy.
        """
        
        # Compute the preferences by dividing the action-values by the temperature parameter tau
        preferences = action_values / tau
        
        # Compute the maximum preference across the actions per row (keepdims to allow broadcasting)
        max_preference = np.max(preferences, axis=1, keepdims=True)
        
        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = np.exp(preferences - max_preference)
        
        # Compute the denominator, i.e., the sum over the numerator along the actions axis (keepdims to allow broadcasting)
        sum_of_exp_preferences = np.sum(exp_preferences, axis=1, keepdims=True)
        
        # Compute the action probabilities
        action_probs = exp_preferences / sum_of_exp_preferences
        
        return action_probs
        
    def get_td_error(self, states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
        """
        Args:
            states (Numpy array): The batch of states with the shape (batch_size, state_dim).
            next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
            actions (Numpy array): The batch of actions with the shape (batch_size,).
            rewards (Numpy array): The batch of rewards with the shape (batch_size,).
            discount (float): The discount factor.
            terminals (Numpy array): The batch of terminals with the shape (batch_size,).
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        Returns:
            The TD errors (Numpy array) for actions taken, of shape (batch_size,)
        """
        
        # Note: Here network is the latest state of the network that is getting replay updates. In other words, 
        # the network represents Q_{t+1}^{i} whereas current_q represents Q_t, the fixed network used for computing the 
        # targets, and particularly, the action-values at the next-states.
        
        # Compute action values at next states using current_q network
        # Note that q_next_mat is a 2D array of shape (batch_size, num_actions)
        q_next_mat = current_q.get_action_values(next_states)

        # Compute policy at next state by passing the action-values in q_next_mat to softmax()
        # Note that probs_mat is a 2D array of shape (batch_size, num_actions)
        probs_mat = self.softmax(q_next_mat, tau)

        # Compute the estimate of the next state value, v_next_vec.
        # Hint: sum the action-values for the next_states weighted by the policy, probs_mat. Then, multiply by
        # (1 - terminals) to make sure v_next_vec is zero for terminal next states.
        # Note that v_next_vec is a 1D array of shape (batch_size,)
        v_next_vec = np.sum(probs_mat * q_next_mat, axis=1) * (1 - terminals) #Estimate of next state value

        # Compute Expected Sarsa target
        # Note that target_vec is a 1D array of shape (batch_size,)
        target_vec = rewards + discount * v_next_vec #Expected SARSA target


        # Compute action values at the current states for all actions using network
        # Note that q_mat is a 2D array of shape (batch_size, num_actions)
        q_mat = network.get_action_values(states) # Compute action values at the current states for all actions using network
        
        # Batch Indices is an array from 0 to the batch size - 1. 
        batch_indices = np.arange(q_mat.shape[0])

        # Compute q_vec by selecting q(s, a) from q_mat for taken actions
        # Use batch_indices as the index for the first dimension of q_mat
        # Note that q_vec is a 1D array of shape (batch_size)
        q_vec = q_mat[batch_indices, actions] # Compute q_vec by selecting q(s, a) from q_mat for taken actions

        # Compute TD errors for actions taken
        # Note that delta_vec is a 1D array of shape (batch_size)
        delta_vec = target_vec - q_vec # Compute TD errors for actions taken
        
        
        return delta_vec

    def optimize_network(self, experiences, discount, optimizer, network, current_q, tau):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
            discount (float): The discount factor.
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        """
        
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        batch_size = states.shape[0]

        # Compute TD error using the get_td_error function
        # Note that q_vec is a 1D array of shape (batch_size)
        delta_vec = self.get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)

        # Batch Indices is an array from 0 to the batch_size - 1. 
        batch_indices = np.arange(batch_size)

        # Make a td error matrix of shape (batch_size, num_actions)
        # delta_mat has non-zero value only for actions taken
        delta_mat = np.zeros((batch_size, network.num_actions))
        delta_mat[batch_indices, actions] = delta_vec

        # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation

        td_update = network.get_TD_update(states, delta_mat)

        # Pass network.get_weights and the td_update to the optimizer to get updated weights
        weights = optimizer.update_weights(network.get_weights(), td_update)
        
        network.set_weights(weights)