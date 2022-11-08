# Importing Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

# Loading Dataset
df = pd.read_csv('Cleaned-Data.csv')

# Deleting less-important columns
severity_columns = df.filter(like='Severity_').columns
df.drop("Country", axis=1, inplace=True)
df.drop("None_Sympton", axis=1, inplace=True)
df.drop("None_Experiencing", axis=1, inplace=True)
df.drop(severity_columns, axis=1, inplace=True)
df.drop("Age_0-9", axis=1, inplace=True)
df.drop("Age_10-19", axis=1, inplace=True)
df.drop("Age_20-24", axis=1, inplace=True)
df.drop("Age_25-59", axis=1, inplace=True)
df.drop("Age_60+", axis=1, inplace=True)
df.drop("Gender_Female", axis=1, inplace=True)
df.drop("Gender_Transgender", axis=1, inplace=True)
df.drop("Contact_Dont-Know", axis=1, inplace=True)
df.drop("Contact_No", axis=1, inplace=True)

# Creating symptoms_score column
df['Symptoms_Score'] = df.iloc[:, :9].sum(
    axis=1) + df.iloc[:, 10:11].sum(axis=1)  # for notebook 19:20

# Separating data and target variables
X = df.iloc[:, :11].values

y = df.iloc[:, 11:12].values

# Splitting train and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Creating the DTR model
dtr = DecisionTreeRegressor(random_state=0)

# Fitting the model
dtr.fit(X_train, y_train)

# Custom prediction with dtr model
'''symptoms_matrix = [[1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]
final_prediction = dtr.predict(symptoms_matrix).astype(int)*10
fp = final_prediction[0].astype(str)
if symptoms_matrix[0][19] == 1:
  print("Your symptoms match with " + fp + "% of the Covid symptoms.\nAs you have been in contact with a Covid patient, we highly recommend you to have a test and stay isolated unitil the result comes.")
elif symptoms_matrix[0][19] == 0:
  print("Your symptoms match with " + fp + "% of the Covid symptoms.")'''

# Saving model to disk
pickle.dump(dtr, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]]))
