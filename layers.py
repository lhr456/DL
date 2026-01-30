import numpy as np

class BaseLayer:
    def __init__(self):
        self.input_data = None
        self.output_data = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, dA): raise NotImplementedError

class Dense(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 使用 Xavier 初始化，防止梯度消失
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.B = np.zeros((1, output_size))
        self.dW = None
        self.db = None

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.dot(self.input_data, self.W) + self.B
        return self.output_data

    def backward(self, dA):
        m = self.input_data.shape[0]
        self.dW = np.dot(self.input_data.T, dA) / m
        self.db = np.sum(dA, axis=0, keepdims=True) / m
        return np.dot(dA, self.W.T)

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

    def backward(self, dA):
        return dA * (self.input_data > 0)