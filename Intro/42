# Binning ,discrete,Linear Models,and Trees
# Last section we talked about represent data on the semantics of the data,
# but that is not the only concern
# we also need to concern the model we are trying to apply
# compare the linear models and tree-based models with wave dataset(1feature 1target)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_wave

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)  # fix column number auto calculate the row number
reg1 = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
reg2 = LinearRegression().fit(X, y)
plt.subplots(1, 2, figsize=(16, 9))
plt.subplot(121)
plt.plot(line, reg1.predict(line), label="decision tree")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.subplot(122)
plt.plot(line, reg2.predict(line), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.savefig("421")
# to make linear models more powerful ,we use binning of feature(split feature)
# define a set of size-fixed bins,original datapoint should fall into one of the bins.
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)
# use oneHot to transform the data
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
line_binned = encoder.transform(np.digitize(line, bins=bins))
plt.figure(figsize=(16, 9))
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), '^', label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.savefig("422")
# so, why we use linear model in the first place, if the dataset is very large,
# and binning can be of great help if some feature has a nonlinear contribution to the output
