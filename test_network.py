import torch
import numpy as np
from randomneuralnetwork import forest
import pandas 





def generate_synthetic_data(n_samples=90000000, n_features=50, n_classes=7):
    """
    Generate synthetic data for testing the neural network forest
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate random labels (3 classes)
    y = np.random.randint(0, n_classes, n_samples)
    print("Before tensoring shape of x ",X.shape, X.dtype,X)
    # Convert to tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)    # Use long tensor for CE LOSS but for Hinge and BCE Loss use Float Tensor
    print(f"Here is x tensors {X.shape} and here is y tensors {y}")
    # print(f"Here is x tensors {X} and here is y tensors {y}")

    # Create a simple dataset class
    class Dataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets
            
    return Dataset(X, y)

def main():
    # Neural network configuration
    neural_config = {
        'trees': 3,
        'input': [5, 5, 5],
        'hidden_layers': [4, 5, 3],
        'nodes_in_hidden': {
            0: [126, 56, 20,8],
            1: [126,56,20, 32, 8],
            2: [64, 16, 4]
        },
        'output': [3, 3, 3],
        'dropout': [0.1, 0, 0],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'loss': ['CE Loss', 'CE Loss', 'CE Loss']
    }
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000, n_features=5, n_classes=3)
    print("Here is the data before passing ",data)
    # Create and train the forest
    print("\nInitializing neural network forest...")
    forest_model = forest(neural_config, data)
    
    # Train the forest
    print("\nTraining the forest...")
    forest_model.train_forest(epochs=20, lr=0.01)
    
    # Evaluate the forest
    print("\nEvaluating the forest...")
    forest_model.evaluate()
    
    # Test with new data
    print("\nTesting with new data...")
    test_data = generate_synthetic_data(n_samples=200, n_features=5, n_classes=3)
    forest_model.network.eval()
    
    with torch.no_grad():
        for i, tree in enumerate(forest_model.network.results):
            outputs = tree(test_data.features)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_data.targets).float().mean()
            print(f'Tree {i+1} Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main() 
