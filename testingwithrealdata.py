import numpy as np 
import pandas as pd 
from randomneuralnetwork import forest 
import torch
from sklearn.model_selection import train_test_split

def processing():

    df = pd.read_csv("Heart Prediction Quantum Dataset.csv")

    x = df.iloc[:,:6]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=42)

   
    
    print("After values of x_train ",x_train.values)
    x_train = torch.FloatTensor(x_train.values)
    y_train = torch.FloatTensor(y_train.values) # For BCE Loss and if LOng for other CE LOss Use long tensor for CE LOSS but for Hinge and BCE Loss use Float Tensor
    print("Y tensor :- ",y_train)
    y_train = y_train.float()
    class Dataset:
        def __init__(self,features,targets):
            self.features = features
            self.targets = targets 

    return Dataset(x_train,y_train)


def test_processing():

    df = pd.read_csv("Heart Prediction Quantum Dataset.csv")
    x = df.iloc[:,:6]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=42)

  
    
    x = x.values
    # print("After values of x ",x)
    x_test = torch.FloatTensor(x_test.values)
    y_test = torch.FloatTensor(y_test.values) # For BCE Loss and if LOng for other CE LOss Use long tensor for CE LOSS but for Hinge and BCE Loss use Float Tensor
    y_test = y_test.float()
    class Dataset:
        def __init__(self,features,targets):
            self.features = features
            self.targets = targets 

    return Dataset(x_test,y_test)


def main():

    import random
    
    # Define hyperparameter search space
    n_trees = random.choice([3, 4, 5])
    activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    dropout_options = [0, 0.1, 0.2, 0.3]
    layer_options = [2, 3, 4, 5]
    node_base = [64, 128, 256]

    neural_config = {
        'trees': n_trees,
        'input': [6] * n_trees,
        'hidden_layers': [random.choice(layer_options) for _ in range(n_trees)],
        'nodes_in_hidden': {
            i: [max(4, random.choice(node_base) // (2**j)) 
                for j in range(random.choice(layer_options))]
            for i in range(n_trees)
        },
        'output': [2] * n_trees,
        'dropout': [random.choice(dropout_options) for _ in range(n_trees)],
        'activation': [random.choice(activations) for _ in range(n_trees)],
        'loss': ['BCE Loss'] * n_trees
    }

    data = processing()
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
    test_data = test_processing()
    forest_model.network.eval()
    
    with torch.no_grad():
        for i, tree in enumerate(forest_model.network.results):
            outputs = tree(test_data.features)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_data.targets).float().mean()
            print(f'Tree {i+1} Test Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
