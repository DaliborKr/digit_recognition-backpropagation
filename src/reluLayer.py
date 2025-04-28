from src import layer as l
import numpy as np

class ReluLayer(l.Layer):
    def __init__(self, number_of_neurons, prev_number_of_neurons, dropout_prob):
        super().__init__(number_of_neurons, prev_number_of_neurons)

        self.dropout_prob = dropout_prob
        self.weights = np.random.randn(prev_number_of_neurons, number_of_neurons) * np.sqrt(2 / prev_number_of_neurons)

    def forward(self, x, dropout_enabled, is_train):
        u = super().forward(x)
        if dropout_enabled and is_train:
            dropout_mask = np.random.binomial(1, self.dropout_prob, size = self.y.shape)
            return (np.maximum(0, u) * dropout_mask) / self.dropout_prob
        else:
            return np.maximum(0, u)

    def backward(self, next_layer):
        delta_weight_sum = next_layer.backward_l()
        self.delta_e = delta_weight_sum * (self.y > 0).astype(float)