import torch
import torch.nn as nn
import math 
import random 



'''
neural {
                'trees':3,
                'input':[5,5,5],
                'hidden_layers':[4,5,3],
                'nodes_in_hidden':{0:[126,56,20],1:[56,32,8],2:[64,16,4]},
                'output':[3,3,3],
                'dropout':[0.1,0.2,0],
                'activation':[relu,relu,sigmoid],
                'loss':['BCE Loss','CE Loss','Hinge Loss']     # BCE Loss , CE Loss , KL Loss, Hinge Loss
                }

'''

class RandomNeuralNetwork(nn.Module):
    def __init__(self, neural, data):
        super(RandomNeuralNetwork,self).__init__()
        # Calculate split indices
        n_samples = len(data.features)
        train_size = int(n_samples * 0.8)
        
        # Split features and targets
        self.train_data = type('Dataset', (), {
            'features': data.features[:train_size],
            'targets': data.targets[:train_size]
        })
        
        self.test_data = type('Dataset', (), {
            'features': data.features[train_size:],
            'targets': data.targets[train_size:]
        })
        
        self.results = []  # Store the results of each tree here
        for i in range(neural['trees']):
            self.results.append(neuraltree(
                neural['input'][i],
                neural['hidden_layers'][i],
                neural['nodes_in_hidden'][i],
                neural['dropout'][i],
                neural['output'][i],
                neural['activation'][i],
                neural['loss'][i],
                self.train_data,
                self.test_data
            ))
    

class neuraltree(nn.Module):
    def __init__(self,input,hidden_layers,nodes,dropout,output,activation,loss,train_data,test_data) -> None:
        super().__init__()
        self.input = input 
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self.dropout = dropout
        self.output = output
        self.activation = activation
        self.loss = loss
        self.train_data = train_data
        self.test_data = test_data

        # Create layers
        input_layer = nn.Linear(input,nodes[0])
        hiddenlayers = []
        output_layer = nn.Linear(nodes[-1],output)
        # print("Here is in neural_tree")    
        # Build hidden layers with proper activation
        for i in range(len(nodes)-1):
            hiddenlayers.append(nn.Linear(nodes[i],nodes[i+1]))
            if activation == 'relu':
                hiddenlayers.append(nn.ReLU())
            elif activation == 'sigmoid':
                hiddenlayers.append(nn.Sigmoid())
            elif activation == 'tanh':
                hiddenlayers.append(nn.Tanh())
            
            if dropout > 0:
                hiddenlayers.append(nn.Dropout(dropout))
            
        # Combine all layers
        layers = [input_layer] + hiddenlayers + [output_layer]
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
    

    
class forest():
    def __init__(self, neural, data):
        self.epochs = 20  # Default 20 
        self.network = RandomNeuralNetwork(neural, data)
        self.optimizers = [torch.optim.Adam(tree.model.parameters()) for tree in self.network.results]
        self.loss_functions = self._get_loss_functions(neural['loss'])
        
    def _get_loss_functions(self, loss_types):
        loss_map = {
            'BCE Loss': nn.BCELoss(),
            'CE Loss': nn.CrossEntropyLoss(),
            'Hinge Loss': nn.HingeEmbeddingLoss(),
            'KL Loss': nn.KLDivLoss()
        }
        return [loss_map[loss] for loss in loss_types]
    
    def train_forest(self, epochs=20, lr=0.01):
        print("In train_forest")
        for epoch in range(epochs):
            total_loss = 0
            for i, (tree, optimizer, loss_fn) in enumerate(zip(self.network.results, self.optimizers, self.loss_functions)):
                # Training mode
                print(f"here is the tree {tree}")
                tree.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = tree(self.network.train_data.features)
                # print("Here is the before output of i ",i,"output ",outputs[:10],"size of outputs ",outputs.shape)
                
                # Handle different loss functions
                if isinstance(loss_fn, nn.BCELoss):
                    # For BCE Loss, use sigmoid on outputs and keep targets as float
                    outputs = torch.sigmoid(outputs)
                    targets = self.network.train_data.targets.float()
                    # Use the first column of outputs for binary classification
                    loss = loss_fn(outputs[:, 0], targets)
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    # For CE Loss, use raw outputs and convert targets to long
                    targets = self.network.train_data.targets.long()
                    loss = loss_fn(outputs, targets)
                elif isinstance(loss_fn, nn.HingeEmbeddingLoss):
                    # For Hinge Loss, convert targets to binary (-1 or 1)
                    targets = (self.network.train_data.targets == 1).float() * 2 - 1
                    loss = loss_fn(outputs.squeeze(), targets)
                else:
                    # Default to CrossEntropyLoss
                    targets = self.network.train_data.targets.long()
                    loss = loss_fn(outputs, targets)
                
                # print("before loss function")
                # print("Here is the output of i ",i,"outputs ",outputs[:10],"size of outputs ",outputs.shape)
                # print(f"Loss function {loss_fn}")
                # print(f"Target shape: {targets.shape}")
                # print(f"Target dtype: {targets.dtype}")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Print progress
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Tree {i+1}, Loss: {loss.item():.4f}')
            
            # Average loss for all trees
            avg_loss = total_loss / len(self.network.results)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    
    def evaluate(self):
        self.network.eval()
        with torch.no_grad():
            total_accuracy = 0
            for i, tree in enumerate(self.network.results):
                outputs = tree(self.network.test_data.features)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == self.network.test_data.targets).float().mean()
                total_accuracy += accuracy
                print(f'Tree {i+1} Accuracy: {accuracy:.4f}')
            
            avg_accuracy = total_accuracy / len(self.network.results)
            print(f'Forest Average Accuracy: {avg_accuracy:.4f}')
