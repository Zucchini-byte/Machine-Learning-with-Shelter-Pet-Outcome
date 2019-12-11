import numpy as np
from sklearn import tree, preprocessing
import pandas as pd

# function to convert the string to number of days
def convertToDays(date):
    date = str(date)
    if(type(date) != type("bruh")):
        print ("this is not correct",date)
    elif("weeks" in date or "week" in date):
        return (int(date[0: 2])) * 7
    elif("years" in date or "year" in date):
        return (int(date[0:2])) * 365
    elif("months" in date or "month" in date):
        return (int(date[0:2])) * 30
    else:
        return (int(date[0:2]))

# reads the file
data = pd.read_csv("/Users/zhoucai/Github/ML_Study/data/preprocessed_data.csv")
print(type(data))

# print the file read
print(data.head(20))

# name of the parameters
param_name = "OutcomeType"
param_name1 = "AnimalType"
param_name2 = "AgeuponOutcome"
param_name3 = "Breed"
param_name4 = "Color"
param_name5 = "SexuponOutcome"


# train_data = data.sample(frac=0.8) # sort

# removes all the nan in the parameters
train_data = data[pd.notnull(data[param_name])]
train_data = train_data[pd.notnull(data[param_name1])]
train_data = train_data[pd.notnull(data[param_name2])]
train_data = train_data[pd.notnull(data[param_name3])]
train_data = train_data[pd.notnull(data[param_name4])]
train_data = train_data[pd.notnull(data[param_name5])]

# outputs the data so far
data_table = pd.DataFrame({
    'Outcome': train_data[param_name],
    'AnimalType': train_data[param_name1],
    'AgeUponOutcome': train_data[param_name2],
    'Breed': train_data[param_name3],
    'Color': train_data[param_name4],
    "SexuponOutcome": train_data[param_name5]
})

print(data_table.head(10))

# converts all the string dates to number of days as ageConvertedList
bruhList = train_data[param_name2].values.tolist()
ageConvertedList = []
for x in range(len(bruhList)):
    if(type(bruhList[x]) != type("String")):
        print(bruhList[x])
    else:
        ageConvertedList.append(convertToDays(bruhList[x]))

train_data[param_name2] = ageConvertedList




# training_result = processed_train_data[param_name].values  # outputs a ndarrays

"""
print(data[param_name1].value_counts())
label_Breed = (data[param_name3].value_counts())
label_Breed.to_csv("/Users/zhoucai/Github/ML_Study/data/Breed_Label_Counts.csv", encoding='utf-8', index=False)
"""

#print(data[param_name4].value_counts())

print("_________________HEre_____")

# breed array printing
print(data[param_name3].value_counts().unique())
print(data[param_name3].value_counts().to_string())
label_array = np.arange(len(data[param_name3].value_counts()))

print(data[param_name].value_counts())

f = open('/Users/zhoucai/Github/ML_Study/data/breed_array.txt','w')
f.write(data[param_name3].value_counts().to_string())
f.close()

breed_label_table = pd.DataFrame({
"Breed Name": (data[param_name3].value_counts(normalize=True)),
"Converted number": label_array
})

breed_label_table.to_csv("/Users/zhoucai/Github/ML_Study/data/Breed_Label_table.csv", encoding='utf-8', index=False)

# Outcome type array printing 0
f = open('/Users/zhoucai/Github/ML_Study/data/outcome_array.txt','w')
f.write(data[param_name].value_counts().to_string())
f.close()
label_array = np.arange(len(data[param_name].value_counts()))
outcome_label_table = pd.DataFrame({
"Outcome": (data[param_name].value_counts(normalize=True)),
"Converted number": label_array
})

outcome_label_table.to_csv("/Users/zhoucai/Github/ML_Study/data/Outcome_Label_table.csv", encoding='utf-8', index=False)

#Animal Type printing 1
f = open('/Users/zhoucai/Github/ML_Study/data/animalType_array.txt','w')
f.write(data[param_name1].value_counts().to_string())
f.close()
label_array = np.arange(len(data[param_name1].value_counts()))
animalType_table = pd.DataFrame({
"Animal Type": (data[param_name1].value_counts(normalize=True)),
"Converted number": label_array
})

animalType_table.to_csv("/Users/zhoucai/Github/ML_Study/data/AnimalType_table.csv", encoding='utf-8', index=False)
# Color printing 4

f = open('/Users/zhoucai/Github/ML_Study/data/color_array.txt','w')
f.write(data[param_name4].value_counts().to_string())
f.close()
label_array = np.arange(len(data[param_name4].value_counts()))
color_label_table = pd.DataFrame({
"Animal Color": (data[param_name4].value_counts()),
"Converted number": label_array
})

color_label_table.to_csv("/Users/zhoucai/Github/ML_Study/data/color_Label_table.csv", encoding='utf-8', index=False)

# Sex upon outcome  5
f = open('/Users/zhoucai/Github/ML_Study/data/sex_array.txt','w')
f.write(data[param_name5].value_counts().to_string())
f.close()

label_array = np.arange(len(data[param_name5].value_counts()))
sex_label_table = pd.DataFrame({
"Sex upon outcome": (data[param_name5].value_counts(normalize=True)),
"Converted number": label_array
})

sex_label_table.to_csv("/Users/zhoucai/Github/ML_Study/data/sex_Label_table.csv", encoding='utf-8', index=False)

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.

# transform categorical data into numbers

train_data[param_name3] = label_encoder.fit_transform(train_data[param_name3])
train_data[param_name1] = label_encoder.fit_transform(train_data[param_name1])
train_data[param_name4] = label_encoder.fit_transform(train_data[param_name4])
train_data[param_name5] = label_encoder.fit_transform(train_data[param_name5])
train_data[param_name] = label_encoder.fit_transform(train_data[param_name])

# print the Breed table converted to numbers
print(train_data[param_name3].head(10))


# separating the data into train and test
print("Here----------")
print(train_data.shape)

# training data
processed_train_data = train_data.sample(frac=0.8)
print(processed_train_data.shape)

# testing data
processed_test_data = train_data.drop(processed_train_data.index)
print(processed_test_data.shape)

# training data

new_data_table = pd.DataFrame({
    'Outcome': processed_train_data[param_name],
    'AnimalType': processed_train_data[param_name1],
    'AgeUponOutcome': processed_train_data[param_name2],
    'Breed': processed_train_data[param_name3],
    'Color': processed_train_data[param_name4],
    "SexuponOutcome": processed_train_data[param_name5]
})

train_data_table = pd.DataFrame({

'AnimalType': processed_train_data[param_name1],
'AgeUponOutcome': processed_train_data[param_name2],
'Breed': processed_train_data[param_name3],
'Color': processed_train_data[param_name4],
"SexuponOutcome": processed_train_data[param_name5]

})

train_target_table = pd.DataFrame({

'Outcome': processed_train_data[param_name],

})

#  test data
new_test_table = pd.DataFrame({
    'Outcome': processed_test_data[param_name],
    'AnimalType': processed_test_data[param_name1],
    'AgeUponOutcome': processed_test_data[param_name2],
    'Breed': processed_test_data[param_name3],
    'Color': processed_test_data[param_name4],
    "SexuponOutcome": processed_test_data[param_name5]
})

test_data_table = pd.DataFrame({

'AnimalType': processed_test_data[param_name1],
'AgeUponOutcome': processed_test_data[param_name2],
'Breed': processed_test_data[param_name3],
'Color': processed_test_data[param_name4],
"SexuponOutcome": processed_test_data[param_name5]

})

test_target_table = pd.DataFrame({

'Outcome': processed_test_data[param_name],

})



new_data_table.to_csv("/Users/zhoucai/Github/ML_Study/data/Processed_data.csv", encoding='utf-8', index=False)
train_data_table.to_csv("/Users/zhoucai/Github/ML_Study/data/train_data.csv", encoding='utf-8', index=False)
train_target_table.to_csv("/Users/zhoucai/Github/ML_Study/data/train_target.csv", encoding='utf-8', index=False)

new_test_table.to_csv("/Users/zhoucai/Github/ML_Study/data/Processed_test.csv", encoding='utf-8', index=False)
test_data_table.to_csv("/Users/zhoucai/Github/ML_Study/data/test_data.csv", encoding='utf-8', index=False)
test_target_table.to_csv("/Users/zhoucai/Github/ML_Study/data/test_target.csv", encoding='utf-8', index=False)

print(new_data_table.head(10))
print(train_data_table.head(10))
print(train_target_table.head(10))

print(new_data_table.shape)
print(train_data_table.shape)
print(train_target_table.shape)

print(new_test_table.shape)
print(test_data_table.shape)
print(test_target_table.shape)


# processed_train_data = train_data.sample(frac=0.8)
# processed_test_data = train_data.drop(train_data.index)
