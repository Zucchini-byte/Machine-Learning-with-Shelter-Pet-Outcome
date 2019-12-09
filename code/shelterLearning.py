import numpy as np
from sklearn import tree, preprocessing
import pandas as pd
from sklearn import tree

train_data = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/train_data.csv")
train_target = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/train_target.csv")

test_data = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/test_data.csv")
test_target = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/test_target.csv")

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target.head(20))

prediction = pd.DataFrame({
"Prediction": clf.predict(test_data)
})

print(prediction.head(20))

score = clf.score(test_data, test_target)
print(score.flatten())
