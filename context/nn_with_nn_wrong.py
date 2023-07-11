import numpy as np

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return np.exp(x) / (1 + np.exp(x))

class NeuralNetwork:
    def __init__(self):
        # Initialize the weights and biases
        self.weights1 =  np.array([2.74, -1.13])
        self.weights2 = np.array([0.36,0.63])
        self.bias1 = np.array([0.0,0.0])
        self.bias2 = 0.0
        self.num_hidden_nodes = 2
        self.num_inputs = 3
        self.hidden_net = np.zeros((self.num_hidden_nodes, self.num_inputs))
        self.hidden_activation = np.zeros((self.num_hidden_nodes, self.num_inputs))
        self.predicted = np.zeros(self.num_inputs)

    def __forward_propagation(self, x):
        for i in range(len(x)):
            for j in range(self.num_hidden_nodes):
                self.hidden_net[j][i] = self.weights1[j] * x[i] + self.bias1[j]
                self.hidden_activation[j][i] = np.log(1 + np.exp(self.hidden_net[j][i]))
                self.predicted[i] += self.hidden_activation[j][i] * self.weights2[j] 
            self.predicted[i] += self.bias2

    def __backward_propagation(self, x, y, lr):
        dssr = np.zeros(len(x))
        diff_bias1 = np.zeros(self.num_hidden_nodes)
        diff_bias2 = 0.0
        diff_weights1 = np.zeros(self.num_hidden_nodes)
        diff_weights2 = np.zeros(self.num_hidden_nodes)

        for i in range(len(x)):
            dssr[i] = -2 * (y[i] - self.predicted[i])
            self.bias2 += dssr[i]
        for i in range(len(x)):
            for j in range(self.num_hidden_nodes):
                diff_bias1[j] += self.dssr[i] * self.weights2[j] * (1 / (1+np.exp(-1.0 * self.hidden_net[j][i])))
                diff_weights1[j] += diff_bias1[j] * self.hidden_net[j][i]
                diff_weights2[j] += dssr[i] * self.hidden_activation[j][i]

        for j in range(self.num_hidden_nodes):
            self.weights1[j] -= diff_weights1[j] * lr
            self.weights2[j] -= diff_weights2[j] * lr
            self.bias1[j] -= diff_bias1[j] * lr
            self.bias2 -= diff_bias2 * lr
        
    def train(self, x, y, epochs=500):
        for epoch in range(epochs):
            self.__forward_propagation(x)
            self.__backward_propagation(x, y, 0.1)

    def predict(self, x):
        self.__forward_propagation(x)
        return self.predicted

# Inputs and expected outputs
x_train = np.array([0.0, 0.5, 1.0])
y_train = np.array([0, 1, 0])

# Estimate the weights and bisases
neural_network = NeuralNetwork()
neural_network.train(x_train, y_train, epochs=100)

#print(neural_network.hidden_net)
#print(neural_network.hidden_activation)
#print(neural_network.predicted)
#print(neural_network.dssr)
print("---------------------")
print(
    neural_network.weights1,
    neural_network.weights2,
    neural_network.bias1,
    neural_network.bias2,
)

# Tests
x_test = np.array([0.0, 0.5, 1.0])
predictions = neural_network.predict(x_test)
for i in range(len(x_test)):
    print("Input:", x_test[i], "Prediction:", predictions[i])