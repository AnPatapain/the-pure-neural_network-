import numpy as np


def sigmoid(x): return 1 / (1 + np.exp(-x))


def tanh(x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(tanh_output): return 1 - tanh_output ** 2


def sigmoid_derivative(sigmoid_output): return sigmoid_output * (1 - sigmoid_output)


def get_x_y_data(data):
    x = data[:, :-1]
    y = data[:, -1]
    y = y.reshape(-1, 1)
    return x, y


class Data:
    def __init__(self):
        self.data = np.loadtxt("data.csv", delimiter=",")
        np.random.shuffle(self.data)

        # Change label with value -1 to 0 to use sigmoid activation function
        for row in self.data:
            if row[-1] == -1:
                row[-1] = 0

        self.train_size = int(0.7 * len(self.data))
        self.val_size = int(0.15 * len(self.data))
        self.test_size = len(self.data) - self.train_size - self.val_size

        self.trainData = self.get_train_data()

    def get_train_data(self):
        return self.data[:self.train_size, :]

    def get_val_data(self):
        return self.data[self.train_size:self.train_size + self.val_size, :]

    def get_test_data(self):
        return self.data[self.train_size + self.val_size:, :]


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.layer1 = None
        self.output = np.zeros(y.shape)
        self.w1 = np.random.rand(self.input.shape[1], self.input.shape[0])
        self.w2 = np.random.rand(self.y.shape[0], self.y.shape[1])
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def loss(self):
        diff = self.y - self.output
        return np.power(diff, 2)

    def loss_derivative_by_weight_layer1(self):
        """
        Suppose that bias is 0
        We have self.layer1 = sigmoid(self.w1 * self.input)
                self.output = sigmoid(self.w2 * self.layer1)
                loss = (self.y - self.output)^2

        Let u0 = self.y - self.output
            u1 = self.w2 * self.layer1
            u2 = self.w1 * self.input

        chain rule d_Loss/d_w1 = d_Loss/d_u0 * d_u0/d_output * d_output/d_u1 * d_u1/d_layer1 * d_layer1/d_u2 * d_u2/d_w1
        """
        return np.dot(self.input.T,
                      np.dot(2 * (self.y - self.output) * self.activation_derivative(self.output), self.w2.T) * self.activation_derivative(self.layer1))

    def loss_derivative_by_weight_layer2(self):
        # using chain rule d_Loss/d_w2 = d_Loss/d_output * d_output/d_w2
        return np.dot(self.layer1.T, (2 * (self.y - self.output) * self.activation_derivative(self.output)))

    def feed_forward(self, input):
        self.layer1 = self.activation(np.dot(input, self.w1))
        self.output = self.activation(np.dot(self.layer1, self.w2))
        # print('self.output')
        # print(self.output)

    def back_propagation(self):
        delta_w1 = self.loss_derivative_by_weight_layer1()
        delta_w2 = self.loss_derivative_by_weight_layer2()
        # fine-tuning the weight for each layer
        self.w1 += delta_w1
        self.w2 += delta_w2

    # def train(self):
    #     for i in range(15000):
    #         self.feed_forward(self.input)
    #         self.back_propagation()

    def train_network(self, learning_rate, iterations):
        for epoch in range(iterations):
            sum_error = 0
            for i in range(len(self.input)):
                print(self.input[i], self.y[i])

    def predict(self, testData):
        input = testData[:, :-1]
        y = testData[:, -1]
        for i in range(len(testData)):
            layer1 = sigmoid(np.dot(input[i], self.w1))
            output = sigmoid(np.dot(layer1, self.w2))
            error = (y[i] - output) ** 2
            print('test row', i, ': ', 'output=', output, 'y=', y[i], 'error=', error)


def main():
    # dataset = np.array([[2.7810836, 2.550537003, 0],
    #            [1.465489372, 2.362125076, 0],
    #            [3.396561688, 4.400293529, 0],
    #            [1.38807019, 1.850220317, 0],
    #            [3.06407232, 3.005305973, 0],
    #            [7.627531214, 2.759262235, 1],
    #            [5.332441248, 2.088626775, 1],
    #            [6.922596716, 1.77106367, 1],
    #            [8.675418651, -0.242068655, 1],
    #            [7.673756466, 3.508563011, 1]])
    #
    # x_train = np.array(dataset[:, :-1])
    # y_train = np.array(dataset[:, -1])
    # y_train = y_train.reshape(len(y_train), 1)
    # print(y_train)
    # neural_network = NeuralNetwork(x_train, y_train)
    # neural_network.train()
    # print('output')
    # print(neural_network.output)

    data_agent = Data()
    # print(repr(data_agent.get_train_data()))
    x_train, y_train = get_x_y_data(data_agent.get_train_data())
    # print('x_train')
    # print(x_train)
    # print('y_train')
    # print(y_train)
    neural_network = NeuralNetwork(x_train, y_train)
    neural_network.train_network(learning_rate=0.5, iterations=20)


main()
