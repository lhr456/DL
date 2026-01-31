import numpy as np

from layers import BaseLayer


class Sigmoid(BaseLayer):
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        return self.output_data

    def backward(self, dA):
        return dA * (self.output_data * (1 - self.output_data))

class ReLU(BaseLayer):
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.maximum(0, input_data)
        return self.output_data
