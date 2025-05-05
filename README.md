# Random Neural Forest

A novel ensemble learning algorithm that combines the strengths of neural networks and random forests. This project implements a "Random Neural Forest"—an ensemble of randomly structured neural networks (trees) that work together to solve classification problems. The approach leverages the diversity of neural architectures to improve generalization and robustness.

## Features

- **Randomized Neural Trees:** Each tree in the forest is a neural network with randomly chosen architecture and hyperparameters.
- **Flexible Loss Functions:** Supports BCE Loss, Cross Entropy Loss, Hinge Loss, and KL Loss.
- **Customizable Ensemble:** Easily configure the number of trees, layers, nodes, activations, and more.
- **Works with Real and Synthetic Data:** Includes examples for both synthetic data and real-world datasets (e.g., heart disease prediction).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Random-Neural-Forest-Algorithm-
   ```

2. **Install dependencies:**
   - Python 3.7+
   - PyTorch
   - NumPy
   - Pandas
   - scikit-learn

   You can install the required packages using pip:
   ```bash
   pip install torch numpy pandas scikit-learn
   ```

## Usage

### 1. Using the Random Neural Forest with Synthetic Data

You can run the provided test script to see the algorithm in action with synthetic data:

```bash
python test_network.py
```

This script:
- Generates random data.
- Configures a random neural forest.
- Trains and evaluates the ensemble.

### 2. Using the Random Neural Forest with Real Data

To use the algorithm with a real dataset (e.g., the provided `Heart Prediction Quantum Dataset.csv`):

```bash
python testingwithrealdata.py
```

This script:
- Loads and preprocesses the dataset.
- Randomly configures the neural forest.
- Trains and evaluates the model.

### 3. Using the Algorithm in Your Own Code

You can import and use the `forest` class from `randomneuralnetwork.py`:

```python
from randomneuralnetwork import forest

# Prepare your data as torch tensors
class Dataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

data = Dataset(features_tensor, targets_tensor)

# Define your neural forest configuration
neural_config = {
    'trees': 3,
    'input': [num_features] * 3,
    'hidden_layers': [4, 5, 3],
    'nodes_in_hidden': {
        0: [126, 56, 20, 8],
        1: [126, 56, 20, 32, 8],
        2: [64, 16, 4]
    },
    'output': [num_classes] * 3,
    'dropout': [0.1, 0, 0],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'loss': ['CE Loss', 'CE Loss', 'CE Loss']
}

# Initialize and train the forest
forest_model = forest(neural_config, data)
forest_model.train_forest(epochs=20, lr=0.01)
forest_model.evaluate()
```

## Configuration Options

- `trees`: Number of neural trees in the forest.
- `input`: List of input sizes for each tree.
- `hidden_layers`: List of hidden layer counts for each tree.
- `nodes_in_hidden`: Dictionary mapping tree index to list of node counts per hidden layer.
- `output`: List of output sizes for each tree.
- `dropout`: List of dropout rates for each tree.
- `activation`: List of activation functions (`'relu'`, `'sigmoid'`, `'tanh'`).
- `loss`: List of loss functions (`'BCE Loss'`, `'CE Loss'`, `'Hinge Loss'`, `'KL Loss'`).

## License

This project is licensed under the Creative Commons CC0 1.0 Universal License. See the [LICENSE](LICENSE) file for details. 
