# Import dataset(Iris)
# Train a classifier
# Predict label for a new flower
# Visualize the tree

# Import Dataset
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()
# print iris.feature_names
# print iris.target_names
print iris.data
# print iris.target[0]

# Put some examples as test data.
test_idx = [0,50,100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
print train_data

#Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train the data
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)
