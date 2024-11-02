# DataScience-Models-and-Deployment : Indtro to Python on Google Colab

## 1) COMMENTS
```py
# Single line comment
print("Hello Google Colab") #first commmand
```

```shell
Hello Google Colab
```
<br>

```py
# Multiline comments
x = 5
y = "Google"
z = "Colab"

print(x)
print(y)
print(z)

"""
5
Google
Colab
"""
```

```shell
5
Google
Colab
```

## 2) VAREABLES
```py
# python basic operators
num1 = 10     # int
num2 = 5.2    # float

num3 = num1 + num2

print(num3)
```

```shell
15.2
```

<br>

```py
# python concatenate strings
firstname = "Kittisak"
lastname = "Hanheam"

fullname = firstname + " " + lastname

print(fullname)
```

```shell
Kittisak Hanheam
```

## 3) MODULES
```py
import pandas as pd #all files
import matplotlib.pyplot as plot
import numpy as np
from sklearn.linear_model import LogisticRegression #specific file(s)
```

## 4) READ FILE
```py
data = pd.read_csv('/content/sample_data/gradingsystem_training.csv')
print(data)
```

![01](/01.png)

<br>

```py
count_data = len(data)
print("Count all rows:" + str(count_data))
```

```shell
Count all rows:30
```

## 5) PLOT GRAPH
```py
import matplotlib.pyplot as plt

x = data['Math']
y = data['CGPA']

plt.scatter(x, y, color='red')
plt.xlabel('Math', fontsize=14)
plt.ylabel('CGPA', fontsize=14)
plt.show()
```

![02](/02.png)

---