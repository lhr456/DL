import numpy as np
from network import Module
from layers import Dense, Sigmoid
from losses import BCE


class XORNet(Module):
    def __init__(self):
        super().__init__()
        # 隐藏层 8 个神经元，给它足够的空间去拟合非线性
        self.fc1 = Dense(2, 8)
        self.act1 = Sigmoid()
        self.fc2 = Dense(8, 1)
        self.act2 = Sigmoid()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        return x


if __name__ == "__main__":
    # 数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = XORNet()
    criterion = BCE()

    # 调优参数
    lr = 0.8  # 提高学习率
    epochs = 10000  # 增加训练次数

    print("正在训练『积木式』自研框架...")
    for epoch in range(epochs + 1):
        loss = model.train_step(X, y, criterion, lr)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5} | Loss: {loss:.6f}")

    # 预测
    print("\n--- 测试结果 ---")
    preds = model.forward(X)
    for i in range(4):
        pred_label = 1 if preds[i] > 0.5 else 0
        print(f"输入: {X[i]} | 预测值: {preds[i][0]:.4f} | 判定: {pred_label}")