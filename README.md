# Least Squares Regression in Python

This repository contains a Jupyter Notebook that demonstrates the implementation of least squares regression using Python. The notebook covers the theoretical background, step-by-step calculations, and practical implementation using both custom functions and the scikit-learn library.

## Notebook Content Overview

### 1. Introduction to Least Squares Regression
- Explanation of the least squares method
- Mathematical derivation of the slope (m) and intercept (c) for the best-fit line

### 2. Data Loading and Preparation
- Import necessary libraries: `numpy`, `pandas`, `matplotlib`
- Load data from a CSV file into a Pandas DataFrame
- Rename DataFrame columns for ease of use

### 3. Calculating Regression Parameters
- Calculate the mean values of X and Y
- Derive the slope (m) and intercept (c) using the least squares method
- Output the calculated slope and intercept

### 4. Plotting the Regression Line
- Generate predicted Y values using the calculated slope and intercept
- Plot the original data points and the regression line

### 5. Error Calculation and Model Evaluation
- Calculate residuals (errors) and plot their distribution using a histogram
- Compute the Residual Sum of Squares (RSS) to evaluate model fit

### 6. Implementation Using scikit-learn
- Initialize and fit a `LinearRegression` model using scikit-learn
- Extract and print the model parameters (slope and intercept)
- Generate predictions using the fitted model and plot the results
- Assess model accuracy using common error metrics: RSS, Mean Squared Error (MSE), and R-squared

## Key Functions and Code Snippets

### Data Loading
```python
df = pd.read_csv('https://github.com/Explore-AI/Public-Data/blob/master/exports%20ZAR-USD-data.csv?raw=true', index_col=0)
df.columns = ['Y', 'X']
X = df.X.values
Y = df.Y.values
```

### Calculating Regression Parameters
```python
x_bar = np.mean(X)
y_bar = np.mean(Y)
m = sum((X - x_bar) * (Y - y_bar)) / sum((X - x_bar) ** 2)
c = y_bar - m * x_bar
print("Slope = ", m)
print("Intercept = ", c)
```

### Plotting the Regression Line
```python
y_gen = m * df.X + c
plt.scatter(df.X, df.Y)
plt.plot(df.X, y_gen, color='red')
plt.show()
```

### Error Calculation and Model Evaluation
```python
errors2 = np.array(y_gen - df.Y)
print(np.round(errors2, 2))
plt.hist(errors2)
plt.show()
print("Residual sum of squares:", (errors2 ** 2).sum())
```

### Implementation Using scikit-learn
```python
from sklearn.linear_model import LinearRegression
X = df.X.values[:, np.newaxis]
lm = LinearRegression()
lm.fit(X, df.Y)
m = lm.coef_[0]
c = lm.intercept_
print("Slope:\t\t", m)
print("Intercept:\t", c)
gen_y = lm.predict(X)
plt.scatter(X, df.Y)
plt.plot(X, gen_y, color='red')
plt.ylabel("ZAR/USD")
plt.xlabel("Value of Exports (ZAR, millions)")
plt.show()
print("Residual sum of squares:", ((gen_y - df.Y) ** 2).sum())
```

### Assessing Model Accuracy
```python
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(df.Y, gen_y))
print("Residual sum of squares:", metrics.mean_squared_error(df.Y, gen_y) * len(X))
print('R_squared:', metrics.r2_score(df.Y, gen_y))
```

## Conclusion
The notebook provides a comprehensive guide to understanding and implementing least squares regression in Python. It covers both manual calculations and the use of scikit-learn for efficient model fitting and evaluation. Users can explore, modify, and extend the code to apply least squares regression to their own datasets.

Feel free to contribute to the repository and provide feedback to improve the notebook!
