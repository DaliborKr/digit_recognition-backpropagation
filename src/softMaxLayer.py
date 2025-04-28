from src import layer as l
import numpy as np

class SoftMaxLayer(l.Layer):
    def __init__(self, number_of_neurons, prev_number_of_neurons):
        super().__init__(number_of_neurons, prev_number_of_neurons)
        self.weights = np.random.randn(prev_number_of_neurons, number_of_neurons) * np.sqrt(2 / (number_of_neurons + prev_number_of_neurons))

    def forward(self, x, dropout_enabled, is_train):
        u = super().forward(x)
        exp_u = np.exp(u - np.max(u))
        return exp_u / np.sum(exp_u)

    def backward(self, label):
        d = np.zeros(self.y.shape[1])
        d[label] = 1
        self.delta_e = d - self.y