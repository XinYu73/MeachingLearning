# Interactions and Polynomials
# example first : add slope
# combine the original feature with the binned feature

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_wave
from sklearn.preprocessing import OneHotEncoder

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)
# use oneHot to transform the data
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
X_combined = np.hstack([X, X_binned])
line_binned = encoder.transform(
    np.digitize(line, bins=bins))
line_combined = np.hstack([line, line_binned])  # horizontally stack
# make ture that the test feature is the same as the training feature
reg = LinearRegression().fit(X_combined, y)
plt.figure(figsize=(16, 9))
plt.plot(line, reg.predict(line_combined), label='linear regression combined')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.plot(X[:, 0], y, 'o', c='k')
plt.legend(loc="best")
plt.savefig("431")
# for now the slope are the same,to get a separate slope for each bin
# we add an interaction feature to indicate which bin a data point is in the original feature
X_product = np.hstack([X_binned, X * X_binned])  # what's the point
reg.fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.figure(figsize=(16, 9))
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.plot(line, reg.predict(line_product), label='linear regression product')
plt.plot(X[:, 0], y, 'o', c='k')
plt.legend(loc="best")
plt.savefig("432")

# We can also seek help from polynomials

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
line_poly = poly.transform(line)
reg = LinearRegression().fit(X_poly, y)
plt.figure(figsize=(16, 9))
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.savefig("4223")

# apply what we have learned on Boston Housing dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
# rescale dateset
scale = MinMaxScaler()
boston.data = scale.fit_transform(boston.data)
# extract polynomial features
poly = PolynomialFeatures(degree=2).fit(boston.data)
boston.data = poly.transform(boston.data)
# second-degree term includes("x0x11,x1x3,x0*x0...")
# split data
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
# load weapon
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("{:.3}".format(ridge.score(X_test, y_test)))
# 0.75
