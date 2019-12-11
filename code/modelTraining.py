import numpy as np
from sklearn import tree, preprocessing
import pandas as pd
from sklearn import tree
import graphviz

train_data = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/train_data.csv")
train_target = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/train_target.csv")

test_data = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/test_data.csv")
test_target = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/test_target.csv")

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target.head(20))

prediction = pd.DataFrame({
"Actual Outcome": test_target["Outcome"],
"Prediction": clf.predict(test_data)
})

print(prediction.head(20))

prediction.to_csv("/Users/zhoucai/Github/ML_Study/data/prediction_comparison.csv", encoding='utf-8', index=False)

score = clf.score(test_data, test_target)
print(score.flatten())

(tree.plot_tree(clf.fit(test_data, test_target)) )



# print(dot_data)


dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("/Users/zhoucai/Github/ML_Study/data/shelter")
