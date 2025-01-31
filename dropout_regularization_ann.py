# dropout_regularization_ann
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('C:\ML\DeepLearning\sonar_dataset.csv',header=None)
print(data)
#checking Null values
print(data.isnull().sum())
#Check the columns Name
print(data.columns)
print(data.iloc[61])
#check the unique value for output column
print(data[60].value_counts())

X = data.drop(60, axis=1)
y = data[60]
y.head()

y = pd.get_dummies(y,drop_first=True).astype('int32') # R---> 1, M-----> 0 
print(y.value_counts())

#Train Test Split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10)

import tensorflow as tf 
from tensorflow import keras 

model = keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation ='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=8)
print('Test evaluation is:\n',model.evaluate(X_test,y_test))

y_predict = model.predict(X_test).reshape(-1)
y_predict = np.round(y_predict)
print('Predicted :\n',np.round(y_predict.reshape(-1))[:10])
print('Actual :\n',y_test[:10])
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_predict))

#Using dropout 
modeld = keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation ='sigmoid')
])

modeld.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modeld.fit(X_train, y_train, epochs=100, batch_size=8)
print('Test evaluation is:\n',modeld.evaluate(X_test,y_test))

y_predict = modeld.predict(X_test).reshape(-1)
y_predict = np.round(y_predict)
print('Predicted :\n',np.round(y_predict.reshape(-1))[:10])
print('Actual :\n',y_test[:10])
from sklearn.metrics import confusion_matrix,classification_report

print('Classification report after using dropout layers :\n',classification_report(y_test,y_predict))