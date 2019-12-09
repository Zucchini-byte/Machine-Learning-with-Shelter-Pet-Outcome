# Add project root folder to module loading paths.
import sys
sys.path.append('/Users/zhoucai/Github/homemade-machine-learning')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import custom linear regression implementation.
from sklearn.linear_model import LinearRegression

# Load the data.
data = pd.read_csv('/Users/zhoucai/Github/homemade-machine-learning/data/world-happiness-report-2017.csv')
print(type(data))

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# Split training set input and output.
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

# Split test set input and output.
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

lr = LinearRegression()

lr.fit(x_train, y_train)
slope = lr.coef_

intercept = lr.intercept_

predictions = lr.predict(x_test)
prediction_table = pd.DataFrame({
'Test vaue' : x_test.flatten(),
'Predicted value': predictions.flatten(),
'True test value': y_test.flatten(),
'Difference between the value': (y_test - predictions).flatten()
})

print(np.mean(y_test - predictions))

prediction_table.head(10)
print ("formula: y = {0}x + {1}".format(slope, intercept))
print(prediction_table)
