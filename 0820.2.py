import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.model_selection import KFold
from  sklearn.metrics import mean_squared_error

from  sklearn.model_selection import cross_val_score

header = ['CRIM','ZN', 'INDUS','CHAS','NOX', 'RM','AGM','DIS','RAD','TAX','RTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('./data/3.housing.csv',delim_whitespace=True,names=header)

array = data.values
X = array[:, 0:13]
Y = array[:, 13]
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3)
model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)


plt.clf()

plt.figure(figsize= (15,6))
plt.scatter(range(len(Y_test[:15])), Y_test[:15], color= 'blue', label= 'Actual Values', marker='o')
plt.scatter(range(len(y_pred[:15])), y_pred[:15], color='red', label='Predicted',marker='*')

plt.title("height and weight")
plt.xlabel("height")
plt.ylabel("weight")
plt.legend()
plt.show()

# mean_squared_error(Y_test,y_pred)
fold = KFold(n_splits=5)
acc = cross_val_score(model,X,Y, cv= fold, scoring= "neg_mean_squared_error")

# print(acc.mean())
print(model.coef_,model.intercept_)