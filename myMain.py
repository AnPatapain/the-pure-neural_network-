import numpy as np


def sigmoid(x): return 1 / (1 + np.exp(-x))


def tanh(x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(tanh_output): return 1 - tanh_output ** 2


def sigmoid_derivative(sigmoid_output): return sigmoid_output * (1 - sigmoid_output)



class Data:
    def __init__(self):
        self.data = np.loadtxt("data.csv", delimiter=",")
        np.random.shuffle(self.data)
        # Change label with value -1 to 0 to use sigmoid activation function
        for row in self.data:
            if row[-1] == -1:
                row[-1] = 0

        # print(self.data)

        self.train_size = int(0.7 * len(self.data))
        self.val_size = int(0.15 * len(self.data))
        self.test_size = len(self.data) - self.train_size - self.val_size

        self.trainData = self.get_train_data()

    def get_x_y_data(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        y = y.astype(int)
        return x, y

    def get_train_data(self):
        return self.data[:self.train_size, :]

    def get_val_data(self):
        return self.data[self.train_size:self.train_size + self.val_size, :]

    def get_test_data(self):
        return self.data[self.train_size + self.val_size:, :]


class Neurone:
    def __init__(self, weight):
        self.weight = weight
        self.output = None
        self.delta = None

    def set_output(self, output): self.output = output

    def set_delta(self, delta): self.delta = delta


class Layer:
    def __init__(self, neurones):
        self.neurones = neurones


def transfer(weight, row):
    bias = weight[-1]
    for i in range(len(weight) - 1):
        bias += weight[i] * row[i]
    return bias


def loss(expected, output):
    return (expected - output)**2


class NeuralNetwork:
    def __init__(self, data, n_input, n_hidden, n_output):
        # Data that feed to network. n_input, n_hidden, n_output is the number of nodes in
        # input, hidden and output layer
        self.data = data
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Network is array of layer. We use two layers. Hidden layer and output layer
        self.hidden_layer = Layer([Neurone(np.random.rand(n_input+1)) for i in range(n_hidden)])
        self.output_layer = Layer([Neurone(np.random.rand(n_hidden+1)) for i in range(n_output)])
        self.layers = [self.hidden_layer, self.output_layer]

        # Activation function and its derivative. Easy to change between sigmoid and tanh
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def feed_forward(self, row):
        # self.layer1 contains 2 neurones of hidden layer and their output
        hidden_layer_outputs = []
        for neurone in self.hidden_layer.neurones:
            output = self.activation(transfer(neurone.weight, row))
            neurone.set_output(output)
            hidden_layer_outputs.append(output)

        # self.output contains 2 neurones of output layer and their output
        for neurone in self.output_layer.neurones:
            neurone.set_output(self.activation(transfer(neurone.weight, hidden_layer_outputs)))

        return [neurone.output for neurone in self.output_layer.neurones]

    def back_propagation(self, expected):
        # We go backwardly, from output layer back to hidden layer to fine-tune w for both layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = []

            # layer is output layer
            if i == len(self.layers)-1:
                for j in range(len(self.output_layer.neurones)):
                    errors.append(self.output_layer.neurones[j].output - expected[j])

            # layer is hidden layer
            else:
                for j in range(len(self.hidden_layer.neurones)):
                    error = 0.0
                    for neurone in self.output_layer.neurones:
                        error += neurone.weight[j] * neurone.delta
                        errors.append(error)

            # Calculate delta for each neurone in layer
            for j in range(len(layer.neurones)):
                layer.neurones[j].delta = errors[j] * self.activation_derivative(layer.neurones[j].output)

    def update_weight(self, row, learning_rate):
        """
        :param row: the row that caused error
        :param learning_rate: How much to change the weight
        :return: None
        """
        for layer in self.layers:
            inputs = row[:-1]
            if layer is self.output_layer:
                inputs = [neurone.output for neurone in self.hidden_layer.neurones]

            for neurone in layer.neurones:
                for j in range(len(inputs)):
                    neurone.weight[j] -= learning_rate * neurone.delta * inputs[j]
                # update the bias
                neurone.weight[-1] -= learning_rate * neurone.delta

    def train(self, learning_rate, iterations):
        # self.feed_forward(self.input[0])
        # self.print_network()
        for epoch in range(iterations):
            sum_error = 0
            for row in self.data:
                outputs = self.feed_forward(row)
                expected = [0 for i in range(self.n_output)]
                expected[row[-1]] = 1
                sum_error += sum([loss(expected[i], outputs[i]) for i in range(len(expected))])
                self.back_propagation(expected)
                self.update_weight(row, learning_rate)
            print('>iteration=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))

    def predict(self, row):
        outputs = self.feed_forward(row)
        return outputs.index(max(outputs))

    def print_network(self):
        print("hidden layer")
        for neuron in self.hidden_layer.neurones:
            print('weight=', neuron.weight, 'output=', neuron.output, 'delta=', neuron.delta)

        print("output layer")
        for neuron in self.output_layer.neurones:
            print('weight=', neuron.weight, 'output=', neuron.output, 'delta=', neuron.delta)

            
def main():
    dataset = [[7.2, 3.2, 6., 1.8, 0],
               [5., 3.5, 1.3, 0.3, 1],
               [7.7, 2.8, 6.7, 2., 0],
               [6.8, 3., 5.5, 2.1, 0],
               [5.4, 3.9, 1.3, 0.4, 1],
               [5.5, 4.2, 1.4, 0.2, 1],
               [4.4, 2.9, 1.4, 0.2, 1],
               [5.3, 3.7, 1.5, 0.2, 1],
               [5., 3.4, 1.5, 0.2, 1],
               [5.1, 3.8, 1.9, 0.4, 1],
               [6.7, 3.3, 5.7, 2.1, 0],
               [5.7, 4.4, 1.5, 0.4, 1],
               [4.9, 2.5, 4.5, 1.7, 0],
               [7.1, 3., 5.9, 2.1, 0],
               [5.2, 3.5, 1.5, 0.2, 1],
               [6.3, 3.3, 6., 2.5, 0],
               [4.7, 3.2, 1.3, 0.2, 1],
               [4.8, 3.1, 1.6, 0.2, 1],
               [4.3, 3., 1.1, 0.1, 1],
               [6.5, 3., 5.2, 2., 0],
               [6., 3., 4.8, 1.8, 0],
               [4.8, 3.4, 1.9, 0.2, 1],
               [6.3, 2.8, 5.1, 1.5, 0],
               [7.6, 3., 6.6, 2.1, 0],
               [4.7, 3.2, 1.6, 0.2, 1],
               [7.7, 3., 6.1, 2.3, 0],
               [5., 3.6, 1.4, 0.2, 1],
               [6.4, 2.8, 5.6, 2.2, 0],
               [7.7, 2.6, 6.9, 2.3, 0],
               [6.3, 2.5, 5., 1.9, 0],
               [4.6, 3.4, 1.4, 0.3, 1],
               [6.3, 2.9, 5.6, 1.8, 0],
               [6.4, 3.2, 5.3, 2.3, 0],
               [6.5, 3.2, 5.1, 2., 0],
               [5.1, 3.4, 1.5, 0.2, 1],
               [7.9, 3.8, 6.4, 2., 0],
               [6.5, 3., 5.5, 1.8, 0],
               [4.6, 3.1, 1.5, 0.2, 1],
               [4.6, 3.2, 1.4, 0.2, 1],
               [5.8, 2.7, 5.1, 1.9, 0],
               [5.9, 3., 5.1, 1.8, 0],
               [4.9, 3.1, 1.5, 0.1, 1],
               [7.4, 2.8, 6.1, 1.9, 0],
               [5.4, 3.4, 1.7, 0.2, 1],
               [5., 3., 1.6, 0.2, 1],
               [5.8, 2.8, 5.1, 2.4, 0],
               [7.2, 3., 5.8, 1.6, 0],
               [6.7, 3.1, 5.6, 2.4, 0],
               [5., 3.5, 1.6, 0.6, 1],
               [5.6, 2.8, 4.9, 2., 0],
               [5.8, 4., 1.2, 0.2, 1],
               [6.3, 3.4, 5.6, 2.4, 0],
               [5.5, 3.5, 1.3, 0.2, 1],
               [5.1, 3.5, 1.4, 0.2, 1],
               [4.4, 3., 1.3, 0.2, 1],
               [5.4, 3.4, 1.5, 0.4, 1],
               [6.9, 3.2, 5.7, 2.3, 0],
               [6.9, 3.1, 5.4, 2.1, 0],
               [4.9, 3., 1.4, 0.2, 1],
               [5., 3.3, 1.4, 0.2, 1],
               [4.9, 3.1, 1.5, 0.1, 1],
               [4.6, 3.6, 1., 0.2, 1],
               [5.7, 2.5, 5., 2., 0],
               [6.1, 2.6, 5.6, 1.4, 0],
               [5.8, 2.7, 5.1, 1.9, 0],
               [6.1, 3., 4.9, 1.8, 0],
               [5.2, 3.4, 1.4, 0.2, 1],
               [6.7, 3.3, 5.7, 2.5, 0],
               [7.2, 3.6, 6.1, 2.5, 0],
               [6.3, 2.7, 4.9, 1.8, 0]]

    datatest = [[4.8, 3.1, 1.6, 0.2, 1],
                [4.4, 3.2, 1.3, 0.2, 1],
                [5.4, 3.9, 1.3, 0.4, 1],
                [6.3, 3.4, 5.6, 2.4, 0],
                [5.3, 3.7, 1.5, 0.2, 1],
                [6.8, 3. , 5.5, 2.1, 0],
                [7.7, 2.6, 6.9, 2.3, 0],
                [6.9, 3.1, 5.1, 2.3, 0],
                [7.2, 3.6, 6.1, 2.5, 0],
                [6.5, 3. , 5.5, 1.8, 0],
                [5.4, 3.4, 1.7, 0.2, 1],
                [5.1, 3.5, 1.4, 0.2, 1],
                [5.1, 3.5, 1.4, 0.3, 1],
                [6. , 2.2, 5. , 1.5, 0],
                [4.8, 3. , 1.4, 0.3, 1]]

    nn = NeuralNetwork(dataset, n_input=len(dataset[0])-1, n_hidden=2, n_output=2)

    nn.train(learning_rate=0.5, iterations=1500)

    print("FINISH TRAINING MODEL. NETWORK AFTER TRAIN")
    nn.print_network()

    print("START TESTING")
    passed = 0
    for row in datatest:
        prediction = nn.predict(row)
        if row[-1] == prediction:
            passed += 1
        print('Expected=%d, Got=%d' % (row[-1], prediction))

    print("Passed", (passed/len(datatest))*100, "%", "test")


main()
