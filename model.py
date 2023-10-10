# Importing Analysis Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for saving the model
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error

df = pd.read_csv('fraud_dataset_example.csv')
print(df.head(5))

# Label encoding for 'type' column
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
df['type_code'] = l.fit_transform(df['type'])

# Dropping isFlaggedFraud since it is false in all rows and also dropping 'type' since we now have type_code
X = df.drop(['isFlaggedFraud', 'isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = df['isFraud']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, lr_pred))
print('MSE:', metrics.mean_squared_error(y_test, lr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))
print(classification_report(y_test, lr_pred))

# k - nearest neighbors
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X, y)
knc_pred = knc.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, knc_pred))
print('MSE:', metrics.mean_squared_error(y_test, knc_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knc_pred)))

print(classification_report(y_test, knc_pred))

# random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))
print('MSE:', metrics.mean_squared_error(y_test, rf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))

print(classification_report(y_test, rf_pred))

print('Model Performance')
accuracy = metrics.accuracy_score(y_test, rf_pred)
print("Test Accuracy: ", accuracy * 100, "%")

print('Train Accuracy:\n {}\n'.format(rf.score(X_train, y_train)))

# saving the model
pickle.dump(knc, open('model.pkl','wb'))