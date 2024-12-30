# Building_SVR_Model
Refer to Feature scaling and SVR model to understand concepts.

## Building the Support Vector Regression Model

### Dataset
The position is encoded to numbers in the level column, hence we will not be applying OneHotEncoder method

_Data Source: SuperDataScience Dataset_

|Position	|Level	|Salary|
|---------|-------|-------|
|Business Analyst|	1|	45000|
|Junior Consultant|	2	|50000|
|Senior Consultant|	3|	60000|
|Manager|	4	|80000|
|Country Manager	|5	|110000|
|Region Manager|	6	|150000|
|Partner|	7	|200000|
|Senior Partner|	8	|300000|
|C-level|	9	|500000|
|CEO|	10	|1000000|
---
## Python codes for Support Vector Regression (SVR)
---

## Importing the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Importing the dataset
```python
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

## Print (X)
```python
print(X)
```

## Print (y)
### Reshaping the y 
> Reasons for reshaping include the following:
> + When you print y after split, you will see it is a 1 D array.
> + The Standardization class of feature scaling requires a 2D array, hence the conversion
> + y= y.reshape(number_of_rows, number_of_column), since we do not want to count rows in y, we use len(y) to do the counting automatically
> + Therefore, we have: y = y.reshape(len(y), 1)
```python
prrint(y)

# Reshape y to appear vertically as default comes horizontally
y = y.reshape(len(y),1)
```

```python
y = y.reshape(len(y), 1)
print(y)
```

## Feature Scaling
_In case a column has a value of 0 and 1, no need to apply feature scaling in that column because it is already in the range of 0 and 1 that feature scaling is trying to achieve. If we do not apply, the value of level which is lower compared to salaries will be neglected by the SVR model_
> Hence, feature scaling is not applied to:
> + Columns with 0 and 1
> + dummy variables as they are already 0 and 1

> Apply feature scaling:
> + Apply feature scaling after split of training and test set, so the test set act as a new dataset for testing
> + Apply feature scaling when the columns have high magnitudes or values over each other as seen in level and salaries above.

## What is inverse transformation of feature scaling?
_Remember you changed the scales or values of the columns to range of 0 and 1, to return the values to the original forms is inverse transformation of feature scaling_
> The inverse transformation of feature scaling refers to the process of converting scaled data back to its original range or scale. This is particularly useful when you need to interpret results or predictions made on scaled data in their original context.

## When Is Inverse Transformation Used?
> + After training a machine learning model to interpret predictions in the original scale.
> + To revert preprocessed data for reporting or visualization.
> + When reconstructing original data from scaled inputs for real-world application.

```python
from sklearn.preprocessing import StandardScaler
# we create two instance as the columns do not have similar dataset
# mean instance computation perfect for Salary (y)
# but not same for level (X)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X) 
y = sc_y.fit_transform(y)

prrint(x)
# See the values are within the range of -3 to +3 which is the range for Standardization
print(y)
```

## Training the SVR model on the whole dataset
## The Gaussian Radial Basis Function (RBF) Kernel
> The Gaussian Radial Basis Function (RBF) Kernel is one of the most commonly used kernels in Support Vector Machines (SVMs) and other kernel-based machine learning algorithms. It is particularly useful for handling non-linear problems by mapping data points into a higher-dimensional space where they become linearly separable.
> This helps make non-learn points 'appear linear'...
> 
+ [Click to view formula for The Gaussian RBF Kernel](https://ibb.co/znFHbXR)
+ [View the plot of that Kernel Here, click to view](https://ibb.co/2SZ85r7)
```python
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
```

## Predicting a new result
```python
# It will take the scale values which was already done for sc_y
# Because, you want to reverse the scaling of output y, you call sc_y
# to avoid format error, you add .reshape(-1, 1)
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))
```

## Visualising the SVR results
```python
# displaying the real values
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
# plot the linear line
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('SVR plot')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

## Python codes results
[You can view the code results in the terminal here. Click to view](https://colab.research.google.com/drive/1ZSrMS4mI1Q4RDC6sZzvivC7oriJzKBRz#scrollTo=v9NK3KIpTk_H)
