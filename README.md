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
prrint(x)
```

## Importing the dataset
```python
prrint(x)
```

## Print (X)
```python
prrint(x)
```

## Print (y)
```python
prrint(y)

# Reshape y to appear vertically as default comes horizontally
y = y.reshape(len(y),1)
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

```python
prrint(x)
```

## Print (X)
```python
prrint(x)
```

## Print (y)
```python
prrint(y)
```

## Training the SVR model on the whole dataset
```python
prrint(y)
```

## Predicting a new result
```python
prrint(y)
```

## Visualising the SVR results
```python
prrint(y)
```
