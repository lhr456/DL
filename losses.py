import numpy as np

class MSE:
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    def grad(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class BCE:
    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def grad(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size