# DataScience-Models-and-Deployment : Chapter-1 Regression Model for Predicting Students Performance

## 1) IMPORT LIBRARIES
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## 2) READ DATASET
```py
dataset = pd.read_csv('/content/gradingsystem_training.csv')
dataset.head(10)
```

![01](/01.png)

## 3) Exploratory Data Analysis
```py
x = dataset['Math']
y = dataset['CGPA']

plt.scatter(x, y, color='red')
plt.xlabel('MATH', fontsize=14)
plt.ylabel('CGPA', fontsize=14)
plt.show()
```

![02](/02.png)

```shell
x = dataset.iloc[:, 2].values.reshape(-1, 1)
y = dataset.iloc[:, 3].values.reshape(-1, 1)
```

## 4) DATA SPLITTING
```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)
```

## 5) CREATE MODEL
```py
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
```

![03](/03.png)

### LINEAR REGRESSION EQUATION

Y = a + bX + error

```py
y_predicted = regression_model.predict(x_test)
y_predicted

y_test
```

```shell
array([[2.5],
       [2. ],
       [3. ],
       [2.3],
       [2.8],
       [2.8],
       [2.5],
       [2.4],
       [2.3]])
```

<br>

```py
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_predicted]})
print(df)
```

```shell
                                              Actual  \
0  [[2.5], [2.0], [3.0], [2.3], [2.8], [2.8], [2....   

                                           Predicted  
0  [[2.675936945401399], [2.1400381041862406], [2...  
```

## 6) MODEL EVALUATION

```py
rmse = mean_squared_error(y_test, y_predicted)
r2_score = r2_score(y_test, y_predicted)
```

```py
print('The intercept is : ', regression_model.intercept_)
print('The coefficient is : ', regression_model.coef_)
print('The rmse is : ', rmse)
print('The r2_score is : ', r2_score)
```

```shell
The intercept is :  [1.64536225]
The coefficient is :  [[0.04122299]]
The rmse is :  0.05792675297327944
The r2_score is :  0.32195563716248055
```

## 7) PREDICTION RESULTS

Y = a + bX + error

```py
# Case 1 - Math score = 30
X = 30

def predicatCGPAscore():
  a = 1.64536225
  b = 0.04122299
  error = 0
  y = a + np.sum(b*X) + 0
  print(y)

predicatCGPAscore()
```

```shell
2.88205195
```

<br>

```py
# Case 2 - Math score = 40
X = 40

def predicatCGPAscore():
  a = 1.64536225
  b = 0.04122299
  error = 0
  y = a + np.sum(b*X) + 0
  print(y)

predicatCGPAscore()
```

```shell
3.29428185
```

## 8) VISUALIZATION
```py
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression_model.predict(x_train), color='blue')
plt.title("Grading System")

plt.xlabel("Math score")
plt.ylabel("CGPA")
plt.show()
```

![04](/04.png)

---