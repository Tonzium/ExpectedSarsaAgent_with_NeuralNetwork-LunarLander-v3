import matplotlib.pyplot as plt
import numpy as np

weights_path = "data/weights/weights_expected_sarsa_agent.npy"

# Load weights
weights = np.load(weights_path, allow_pickle=True)

def visualize_weights(weights):
    for i, layer in enumerate(weights):
        print(f"Layer {i}:")
        if isinstance(layer, dict):
            # Explore or visualize dictionary-structured weights
            for key, value in layer.items():
                if isinstance(value, np.ndarray):
                    plt.hist(value.flatten(), bins=50)
                    plt.title(f'Layer {i} - {key} Weight Distribution')
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"Key '{key}' is not an array.")
        elif isinstance(layer, np.ndarray):
            # Direct visualization if the layer itself is an ndarray
            plt.hist(layer.flatten(), bins=50)
            plt.title(f'Layer {i} Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"Layer {i} contains unexpected data type.")

# Call the function with your weights
visualize_weights(weights)