# neural network with iris dataset
## Dependency
*  csv
* numpy 
* math
* random

## Model
__Network__ is array of __layers__. __Layers__ is array of __neurones__. One __Neurone__ has 
array of __weights__ that associates it to the neurones in previous layer. 
There are three layers, input, hidden and output layer. Hidden layer has 
10 neurones, output layer has 3 neurones and input layer has 4 neurones.

Use hot-encoded to encode the output at output layer
## Functions
### Feedforward
Activation function for __hidden layer__ is __Sigmoid__ while activation function
for __output layer__ is __Softmax__.
### Backpropagation
Calculate gradient of weight in weights array of each neurone in each layer.
For neurone in __output layer__ the gradient is __output - expected__. For hidden_layer.neurone[i] in 
hidden layer the gradient is __output_layer.neurones[i].delta * output_layer.neurones[i].weights[i] * sigmoid_derivative(hidden_layer.neurone[i].output)__
### Train
Train with iteration is 100 and the threshold is 90% accuracy.
### Update weight
Update the weight in weights array of each neurone in each layer using the gradient calculated in 
backpropagation function
