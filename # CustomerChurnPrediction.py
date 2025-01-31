# Customer churn prediction
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('C:\ML\DeepLearning\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(data)

#Remove unwanted column customer id
data = data.drop('customerID',axis=1)
print(data)

#To know the datatype of all the columns 
print(data.info())
print('TotalCharges :',data.TotalCharges.values)
print('MonthlyCharges :',data.MonthlyCharges.values)
print(data['TotalCharges'].isnull().sum())
null_rows = data[data['TotalCharges'].isnull()].reset_index(drop=True)
print('Total charges wich has space :\n',null_rows)
pd.to_numeric(data.TotalCharges,errors='coerce') #conver to numeric if error ignore and convert the other values 
                                                 # errors='coerce' converts non-numeric values to NaN
print('Data having null values in Totalcharges columns:\n',data[pd.to_numeric(data.TotalCharges,errors='coerce').isnull()] )# to show only values having null 

print(data.iloc[488]) #iloc means index location only show the one row values. If we see TotalCharges is empty 
print('TotalCharges:\n',data.iloc[488]['TotalCharges']) # to see only this filed values
# to check the unique value for each column
data1 = data[data['TotalCharges'] != ' '] # only storing totalchages having proper values 
data1['TotalCharges'] = pd.to_numeric(data1.TotalCharges) #converting datatype to numeric 
print(data1.isnull().sum())
print(data1.info())

print('Tensure_churn_no:=\n',data1[data1['Churn'] == 'No'].tenure)

Tensure_churn_no = data1[data1['Churn'] == 'No'].tenure
Tensure_churn_Yes = data1[data1['Churn'] == 'Yes'].tenure
plt.hist([Tensure_churn_Yes,Tensure_churn_no],color=['green','red'],label=['churn=Yes','churn=No'])
plt.legend() #this is to show Churn = Yes and Churn = No
plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.title('Customer churn prediction visualization')
plt.show()

MonthlyCharges_churn_no = data1[data1['Churn'] == 'No'].MonthlyCharges
MonthlyCharges_churn_Yes = data1[data1['Churn'] == 'Yes'].MonthlyCharges
plt.hist([MonthlyCharges_churn_Yes,MonthlyCharges_churn_no],color=['green','red'],label=['churn=Yes','churn=No'])
plt.legend() #this is to show Churn = Yes and Churn = No
plt.xlabel('MontlyCharges')
plt.ylabel('Number of customers')
plt.title('Customer churn prediction visualization')
plt.show()
 

def handling_Categorical_data(data1):
    for column in data1.columns:    
        if data1[column].dtype == 'object':
            print(f'{column}:{data1[column].unique()}')

handling_Categorical_data(data1)

data1 = data1.replace('No internet service','No')
data1 = data1.replace('No phone service','No')
handling_Categorical_data(data1)

column_list_to_replace = ['Churn','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                          'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','gender'] 

for column  in column_list_to_replace:
  
     data1[column].replace({'No':0,'Yes':1},inplace=True)    
     if column == 'gender':
        data1[column].replace({'Female':0,'Male':1},inplace=True) 

print(data1.columns)   
 
data1 = pd.get_dummies(data1,columns=['InternetService','Contract','PaymentMethod'])
data1 = data1.astype(int)  # Convert all boolean values to 0 and 1
print(data1.sample(10))   
print(data1.info())   # to know the data types of each column
print(data1)
 
 
''' 
if 'InternetService' in data1.columns:
    print('Yes yes')
    data1 = pd.get_dummies(data1, columns=['InternetService'])
else:
    print("Error: 'InternetService' column not found in DataFrame!")

print( data1.columns)

if 'Contract' in data1.columns:
    data1 = pd.get_dummies(data1,columns=['Contract'])
else:
    print("Error: 'Contract' column not found in DataFrame!")  

if 'PaymentMethod' in data1.columns:
    data1 = pd.get_dummies(data1,columns=['PaymentMethod'])
else:
    print('Error:PaymentMethod is not available in the dataframe')

print('Pring unique values in the data frame is:\n')
handling_Categorical_data(data1)
print( data1.columns)

''' 

column_to_scale = ['MonthlyCharges','TotalCharges','tenure']

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
data1[column_to_scale] = scaler.fit_transform(data1[column_to_scale])

print(data1['TotalCharges'].unique()) # after scaling TotalCharges changed between 0 to 1

X = data1.drop('Churn',axis=1)
y = data1['Churn']
print('Y value is:\n',y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=90)
import tensorflow as tf 
from tensorflow import keras 
# input layers as number of independent columns in the dataset
print(X_train.shape)
model = keras.Sequential([
     keras.layers.Dense(20,input_shape = (X_train.shape[1],),activation='relu'),
     keras.layers.Dense(15,activation='relu'),
     keras.layers.Dense(10,activation='relu'),
     keras.layers.Dense(1,activation ='sigmoid')      
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100)
print('Model evaluation is:\n',model.evaluate(X_test,y_test))

y_predicted = model.predict(X_test)

y_predict =[]
for i in y_predicted:
    if i > 0.5:
        y_predict.append(1)
    else:
        y_predict.append(0) 
    
print('Predicted values :\n',y_predict[:10])   
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_predict))
import seaborn as sn 
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predict)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('truth')
plt.show()