import numpy as np
from sklearn import tree, preprocessing
import pandas as pd
from sklearn import tree
import graphviz

train_data = pd.read_csv("/Users/zhoucai/Github/Python/ML_Study/data/train_data.csv")
train_target = pd.read_csv("/Users/zhoucai/Github/Python/ML_Study/data/train_target.csv")

test_data = pd.read_csv("/Users/zhoucai/Github/Python/ML_Study/data/test_data.csv")
test_target = pd.read_csv("/Users/zhoucai/Github/Python/ML_Study/data/test_target.csv")

# training of the model

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#this will prints the test targets--------heading

#print (test_target.head(20))

#line below prints the outcome of the test target's actual outcome and predicted outcome

prediction = pd.DataFrame({
"Actual Outcome": test_target["Outcome"],
"Prediction": clf.predict(test_data)
})

#print(prediction.head(20))

# prints the prediction to a csv file.  -----

prediction.to_csv("/Users/zhoucai/Github/Python/ML_Study/data/prediction_comparison.csv", encoding='utf-8', index=False)

# prints the accuracy of the model
score = clf.score(test_data, test_target)
print(score.flatten())

repeat = True;
while repeat:
    print("Enter the traits of the pet: (Animal Type, Age(In days), Breed, Color, Gender)")

    print("Animal Type:")
    animalType = input()

    print("Age: ")
    age = input()

    print("Breed: ")
    breed = input()

    print("Color: ")
    color = input()

    print("Gender: ")
    gender =input()

    output = pd.DataFrame(np.array([[animalType,age,breed,color, gender]]), columns=['AnimalType', 'age', 'breed', 'color', 'gender'])

    petOutcome = (clf.predict(output))
    print("--------------------------------------------------")
    if petOutcome == 0:
        print("The animal you entered has the highest chance in getting adopted. ")
    elif petOutcome == 1:
        print("The animal you enter has the highest chance of getting transfered.")
    elif petOutcome == 2:
        print("The animal you enter has the highest chance of getting returned back to the owner.")
    elif petOutcome == 3:
        print("The animal you enter has the highest chance of getting euthanized.")
    elif petOutcome == 4:
        print("The animal you enter has the highest chance of death.")

    print("--------------------------------------------------")
    print("Would you like to enter another animal? (Y/N) ");
    answer = input()
    if answer == "N":
        repeat = False
    elif answer == "n":
        repeat = False;
    print("--------------------------------");
    print("\n\n");







# (tree.plot_tree(clf.fit(test_data, test_target)) )
# print(dot_data)

# outputs the decision tree
#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("/Users/zhoucai/Github/ML_Study/data/shelter")
