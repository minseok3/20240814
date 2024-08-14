import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection  import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

header = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv("./data/2.iris.csv", names=header)
# 데이터 전처리 : Min-Max 전처리
array = data.values
X = array[:, 0:4]
Y = array[:, 4]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 모델 선택 및 분할
model = DecisionTreeClassifier()
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

con = confusion_matrix(Y_pred, Y_test)
print(con)

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring="accuracy")
print(acc)




# 결과(모델 예측값 vs 실제값) 시각화
plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_pred)), Y_pred, color='red', label='Predicted Values', marker='x')

plt.title("Comparison of Actual and Predicted Values")
plt.xlabel("Index")
plt.ylabel("Class (0 or 1)")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')



