# Automatic Feature Selection
# When adding new features or dealing with High-dimensional datasets
# what are good feature??
# Three basic metrics: uni-variate statistics,model-based selection ,iterative selection

# 451Uni-variate Statistics
# consider each feature individually
# value the statistically significant relationship between each feature and the target

# apply on cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
# first we add some noise feature to the original dataset
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
cancer.data = np.hstack([cancer.data, noise])  # horizontally add
select = SelectPercentile(percentile=50)
select.fit(cancer.data, cancer.target)  # do not forget the target
X_selected = select.transform(cancer.data)
print("{}".format(X_selected.shape))
print("{}".format(cancer.data.shape))
# visualize what the chosen features are
plt.figure()
plt.matshow(select.get_support().reshape(1, -1), cmap='gray_r')  # 1 row ,auto-count columns
plt.xlabel("Sample index")
plt.savefig("451")

# 452Model-Based Feature Selection
# the selection uses a supervised machine learning model to judge the importance of each feature
# model: tree-based, linear-model(coefficients)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
select.fit(cancer.data, cancer.target)
X_selected1 = select.transform(cancer.data)
plt.figure()
plt.matshow(select.get_support().reshape(1, -1), cmap='gray_r')  # 1 row ,auto-count columns
plt.xlabel("Sample index")
plt.savefig("452")

# 453Iterative Feature selection
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(cancer.data, cancer.target)
X_selected2 = select.transform(cancer.data)
plt.figure()
plt.matshow(select.get_support().reshape(1, -1), cmap='gray_r')  # 1 row ,auto-count columns
plt.xlabel("Sample index")
plt.savefig("453")
