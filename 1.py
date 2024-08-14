
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/1.salary.csv')
# 독립변수
X = data['Experience Years']
# 종속변수
Y = data['Salary']

X = X.values
X = X.reshape(-1,1)

# 데이터 분할
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3)

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train,Y_train)
model.coef_
model.intercept_

y_pred = model.predict(X_test)


df_y_test = pd.DataFrame(Y_test)

plt.clf()
# 결과값 예측
plt.figure(figsize= (15,6))
plt.scatter(X_test, Y_test, color= 'blue', label= 'Actual Values', marker='o')
plt.plot(X_test, y_pred, color='red', label='Predicted',marker='*')

plt.title("comparaison of Actual and Predicted Values")
plt.xlabel("index")
plt.ylabel("class(years)")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

acc = mean_absolute_error(Y_test,y_pred)
print(acc)