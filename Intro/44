# Unit-variate Nonlinear Transformations
# linear models and neural networks are very tied to the scale and distribution of each feature
# if there is a nonlinear relation , modelling will be tricky.
# Functions like log, exp, sin can help.
# Most models work best when each feature (and in regression also the target) is looselyGaussian  distributed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)
X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
bins = np.bincount(X[:, 0])
plt.figure(figsize=(16, 9))
plt.bar(range(len(bins)), bins, color='b')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.savefig("441")

from sklearn.linear_model import Ridge

X = np.log(X + 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))

# from the previous examples, one can see that
# binning,  polynomials, and  interactions  can have a huge influence on how models perform on a given dataset. T
# This is particularly true  for  less  complex  models
# like  linear  models  and  naive  Bayes  models.
