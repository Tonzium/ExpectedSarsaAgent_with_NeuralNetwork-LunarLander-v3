import numpy as np
from copy import deepcopy
import json

with open("configs/parameters.json", "r") as f:
    network_config = json.load(f)

class ActionValueNetwork:
    """
    Manual neural network implementation
    """
    
    def __init__(self, network_config):
        self.state_dim = network_config["state_dim"]
        self.num_hidden_units_1 = network_config.get("num_hidden_units", 256)
        self.num_hidden_units_2 = network_config.get("num_hidden_units_2", 128)
        self.num_actions = network_config["num_actions"]
        
        self.rand_generator = np.random.RandomState(network_config["seed"])
        
        # Specify self.layer_sizes: [input, hidden1, hidden2, output]
        self.layer_sizes = [self.state_dim, self.num_hidden_units_1, self.num_hidden_units_2, self.num_actions]
        
        
        # Initialize the weights of the neural network
        # self.weights is an array of dictionaries with each dictionary corresponding to 
        # the weights from one layer to the next.
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))
    
    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        # Layer 1
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi0 = np.dot(s, W0) + b0
        x0 = np.maximum(psi0, 0)
        
        # Layer 2
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        psi1 = np.dot(x0, W1) + b1
        x1 = np.maximum(psi1, 0)

        # Output Layer
        W2, b2 = self.weights[2]['W'], self.weights[2]['b']
        q_vals = np.dot(x1, W2) + b2

        return q_vals
    
    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions).
        Returns:
            The TD update (Array of dictionaries)
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        W2, b2 = self.weights[2]['W'], self.weights[2]['b']
        
        # Forward pass for gradients
        psi0 = np.dot(s, W0) + b0
        x0 = np.maximum(psi0, 0)
        dx0 = (psi0 > 0).astype(float)

        psi1 = np.dot(x0, W1) + b1
        x1 = np.maximum(psi1, 0)
        dx1 = (psi1 > 0).astype(float)

        td_update = [dict() for i in range(len(self.weights))]
         
        # Output layer gradients (Layer 2 -> Output)
        v = delta_mat
        td_update[2]['W'] = np.dot(x1.T, v) * 1. / s.shape[0]
        td_update[2]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        
        # Hidden layer 2 gradients (Layer 1 -> Layer 2)
        v = np.dot(v, W2.T) * dx1
        td_update[1]['W'] = np.dot(x0.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        # Hidden layer 1 gradients (Input -> Layer 1)
        v = np.dot(v, W1.T) * dx0
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
                
        return td_update
    
    # Exact solutions to the nonlinear dynamics of learning in deep linear neural networks by Saxe, A et al., 2013
    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)
    
    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)