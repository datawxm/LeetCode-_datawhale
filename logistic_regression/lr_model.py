import numpy as np

# 生成一些示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置列到特征矩阵
X_b = np.c_[np.ones((100, 1)), X]

# 超参数
learning_rate = 0.1
n_iterations = 1000
m = 100

# 初始化权重
theta = np.random.randn(2, 1)

# 梯度下降
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("权重和偏置:", theta)

# 预测
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)

print("预测值:", y_predict)
