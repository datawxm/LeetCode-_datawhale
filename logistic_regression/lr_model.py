import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 创建示例数据集
data = {
    'feature1': [0.1, 0.4, 0.35, 0.8, 0.3, 0.5, 0.7, 0.6, 0.2, 0.9],
    'feature2': [0.2, 0.5, 0.3, 0.9, 0.4, 0.6, 0.8, 0.7, 0.1, 1.0],
    'label': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# 分离特征和标签
X = df[['feature1', 'feature2']]
y = df['label']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('混淆矩阵:')
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred)
print('分类报告:')
print(class_report)
