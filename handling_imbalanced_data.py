
print( ''' Handling imbalanced data in customer churn prediction
            Customer churn prediction is to measure why customers are leaving a business. 
            In this tutorial we will be looking at customer churn in telecom business. 
            We will build a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of our model. 
            We will then handle imbalance in data using various techniques and improve f1-score ''')

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
 
import warnings
warnings.filterwarnings('ignore')
# Load the data

df = pd.read_csv('C:\ML\DeepLearning\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.sample(5))
df.drop('customerID',axis=1,inplace=True)
print(df.sample(5))

print(df.isnull().sum())
print(df.info())
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype('float64')
print(df.info())

def check_unique_val(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            print(f'{column} : {df[column].unique()}')
            
check_unique_val(df)    

df['DeviceProtection'] = df['DeviceProtection'].replace('No internet service','No')
df['OnlineSecurity'] = df['OnlineSecurity'].replace('No internet service','No')
df['StreamingMovies'] = df['StreamingMovies'].replace('No internet service','No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service','No')
df['DeviceProtection'] = df['DeviceProtection'].replace('No internet service','No')
df['TechSupport'] = df['TechSupport'].replace('No internet service','No')
df['StreamingTV'] = df['StreamingTV'].replace('No internet service','No')
df['MultipleLines'] = df['MultipleLines'].replace('No internet service','No')
df['OnlineBackup'] = df['OnlineBackup'].replace('No internet service','No')
print('\n\n')
check_unique_val(df) 

column_list = ['gender','Churn','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']

for column in column_list:
    if column == 'gender':
        df[column] = df[column].replace({'Male':1,'Female':0})
    else:    
       df[column] = df[column].replace({'Yes':1,'No':0})

print(df.sample(10))
check_unique_val(df) 
df = pd.get_dummies(df,columns=['InternetService','Contract','PaymentMethod']).astype('int64')
print(df.sample(10))
print(df.columns)
print(df.info())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
column_to_scale = ['MonthlyCharges','TotalCharges','tenure']

df[column_to_scale] = scaler.fit_transform(df[column_to_scale])
print(df[column_to_scale])

 
print(df.shape)

print(df['Churn'].value_counts())

#Data Visualization
tenure_churn_no = df[df.Churn==0].tenure
tenure_churn_yes = df[df.Churn==1].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()

mc_churn_no = df[df.Churn==0].MonthlyCharges      
mc_churn_yes = df[df.Churn==1].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()

X = df.drop('Churn',axis='columns')
y = testLabels = df.Churn 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

#it is a imbalanced data because class 0 has 4130 and class 1 has 1495
print(y_train.value_counts())
print(y_test.value_counts())
print('Total input columns:\n',len(X_train.columns))

#Build a model (ANN) in tensorflow/keras

#from tensorflow_addons import losses
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report

def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight = weights)
    
    print(model.evaluate(X_test, y_test))
    
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))
    
    return y_preds


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#Mitigating Skewdness of Data
#Method 1: Undersampling
#reference: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
# Class count
count_class_0, count_class_1 = df.Churn.value_counts()
print(count_class_0,count_class_1)

# Divide by class
df_class_0 = df[df['Churn'] == 0]
df_class_1 = df[df['Churn'] == 1]

# Undersample 0-class and concat the DataFrames of both class
df_class_0_under = df_class_0.sample(count_class_1) # class 0 has more values so reducing it as class1 values here ->1869
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:') # Now both classes has sample number of classifications 
print(df_test_under.Churn.value_counts())

X = df_test_under.drop('Churn',axis='columns')
y = df_test_under['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
# Number of classes in training Data
print(y_train.value_counts())
# After undersampling model is trained 
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

# Method2: Oversampling
# Oversample 1-class and concat the DataFrames of both classes
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Churn.value_counts())

X = df_test_over.drop('Churn',axis='columns')
y = df_test_over['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
# Number of classes in training Data
print(y_train.value_counts())

#loss = keras.losses.BinaryCrossentropy()
#weights = -1
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#Method3: SMOTE
#To install imbalanced-learn library use pip install imbalanced-learn command

X = df.drop('Churn',axis='columns')
y = df['Churn']
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

print(y_sm.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
# Number of classes in training Data
print(y_train.value_counts())

y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

#SMOT Oversampling increases f1 score of minority class 1 from 0.57 to 0.81 (huge improvement) Also over all accuracy improves from 0.78 to 0.80

#Method4: Use of Ensemble with undersampling
print(df.Churn.value_counts())

# Regain Original features and labels
X = df.drop('Churn',axis='columns')
y = df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
y_train.value_counts()

df3 = X_train.copy()
df3['Churn'] = y_train
df3.head()

df3_class0 = df3[df3.Churn==0]
df3_class1 = df3[df3.Churn==1]
def get_train_batch(df_majority, df_minority, start, end):
    df_train = pd.concat([df_majority[start:end], df_minority], axis=0)

    X_train = df_train.drop('Churn', axis='columns')
    y_train = df_train.Churn
    return X_train, y_train    
X_train, y_train = get_train_batch(df3_class0, df3_class1, 0, 1495)

y_pred1 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 1495, 2990)

y_pred2 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 2990, 4130)

y_pred3 = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)

y_pred_final = y_pred1.copy()
for i in range(len(y_pred1)):
    n_ones = y_pred1[i] + y_pred2[i] + y_pred3[i]
    if n_ones>1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0
cl_rep = classification_report(y_test, y_pred_final)
print(cl_rep)

