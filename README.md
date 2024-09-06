# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Atchaya.k
RegisterNumber:  212223220011
*/
```
```


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:


# df.head()

![image](https://github.com/user-attachments/assets/3c336cc5-e876-4143-9935-3afaf2c9cab6)

# df.tail()

![image](https://github.com/user-attachments/assets/58b0641a-c413-4272-8436-bc9c58f8cf3a)

# Array value of X

![image](https://github.com/user-attachments/assets/eb606529-b249-47b1-ba1d-d2e6bdf7d367)

# Array value of Y

![image](https://github.com/user-attachments/assets/bf2d15fc-458d-496e-84e5-c473afabdfaa)

# Values of Y prediction

![image](https://github.com/user-attachments/assets/17f1783c-70d6-4b15-bc5f-feed39035283)

# Array values of Y test

![image](https://github.com/user-attachments/assets/8712372c-bfe5-4cc6-a3ff-4dd7cf9c9a15)


# Training set graph

![image](https://github.com/user-attachments/assets/2e133fd9-eb8a-4cf1-b8b1-172c0cd08fe7)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
