import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.model_selection import KFold
from  sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)


# 독립변수
X = data['Weight(Pounds)']
# 종속변수
Y = data['Height(Inches)']
data['Height(Inches)'] = data['Height(Inches)']*2.54
data['Weight(Pounds)'] = data['Weight(Pounds)']*0.453592

array = data.values
array.shape
y= array[:,0]
x= array[:,1]
X = x.reshape(-1,1)
# 데이터 분할
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3)
model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
print(data)

# model.coef_
# model.intercept_
y_pred = model.predict(X_test)

acc = mean_absolute_error(Y_test,y_pred)

plt.clf()
# 결과값 예측
plt.figure(figsize= (15,6))
plt.scatter(X_test[:100], y_pred[:100], color= 'blue', label= 'Actual Values', marker='o')
plt.scatter(X_test[:100], Y_test[:100], color='red', label='Predicted',marker='*')

plt.title("height asnd weight")
plt.xlabel("height")
plt.ylabel("weight")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

acc = mean_absolute_error(Y_test,y_pred)
print(acc)
