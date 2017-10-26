# Use inbuilt Decision Tree Classifier
# Get training data
# Get labels for each one of them
# Train the classifier
# Predict using unseen values

from sklearn import tree
features = [[140,1], [130,1], [150,1], [170,1]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[160,1], [130,1]])
