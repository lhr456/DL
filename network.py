from layers import BaseLayer


class Module:
    def __init__(self):
        self.__dict__['_layers'] = []

    def __setattr__(self, name, value):
        if isinstance(value, BaseLayer):
            self._layers.append(value)
        super().__setattr__(name, value)

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def train_step(self, X, y, criterion, learning_rate):
        # 1. 前向
        y_pred = self.forward(X)

        # 2. 计算起始梯度
        loss_val = criterion(y, y_pred)
        dA = criterion.grad(y, y_pred)

        # 3. 反向传播并原地更新
        for layer in reversed(self._layers):
            dA = layer.backward(dA)
            if hasattr(layer, 'W'):
                layer.W -= learning_rate * layer.dW
                layer.B -= learning_rate * layer.db

        return loss_val