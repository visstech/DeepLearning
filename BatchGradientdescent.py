#Batch Gradient descent 
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt 
data = pd.read_csv('C:\ML\DeepLearning\homeprices_banglore.csv')
print(data
      )
sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()
X_scaled = sx.fit_transform(data.drop('price',axis=1))
print(X_scaled)
#print('price before reshape:\n',sy.fit_transform(data['price']))
#y_scaled = sy.fit_transform(data['price'])
# ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> 
# instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
y_scaled = sy.fit_transform(data['price'].values.reshape(data.shape[0],1))
print('price after reshape:\n',y_scaled.shape)
print(data['price'].shape)

#transpose meaning The transpose of a matrix is obtained by moving the rows data to the column and columns data to the rows
print('Before transpose:\n',X_scaled)
print('After transpose:\n',X_scaled.T)

def batch_gradient_descent(X,y_true,epochs,learning_rate = 0.001):
    
    number_of_features = X.shape[1] # Number of columns 
    w = np.ones(shape=(number_of_features)) # it will assign 1 to each waits w1,w2 
    b = 0 # bias 
    total_samples = X.shape[0] # number of rows 
    
    cost_list = []
    epochs_list = []
   
    for i in range(epochs):
      
        #formula is  y_predicted = W1 * area + W2 * bedrooms + bias;
        y_predicted = np.dot(w,X_scaled.T) + b #np.dot for matrix multiplication each wait values multiplied by area and w2 multiplied with bedrooms
        
        w_grad = -(2/total_samples) * (X.T.dot(y_true - y_predicted))
        b_grad = -(2/total_samples) * np.sum(y_true - y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        cost = np.mean(np.square(y_true - y_predicted))
        
        if i%10 == 0 :
            cost_list.append(cost)
            epochs_list.append(i)
        
           # print(f'waits:{w},bias:{b},cost:{cost},cost list:{cost_list},epochs:{epochs_list}') 

    return w,b,cost,cost_list,epochs_list
    

w,b,cost,cost_list,epochs_list = batch_gradient_descent(X_scaled,y_scaled.reshape(y_scaled.shape[0],),500)
print(f'waits:{w},bias:{b},cost:{cost},cost list:{cost_list},epochs:{epochs_list}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(epochs_list,cost_list)
plt.show()

def predict(area,bedroom,w,b):
    scaled_X = sx.transform([[area,bedroom]])[0]
    scaled_price = w[0]*scaled_X[0] + w[1] *scaled_X[1] + b
    scaled_price = sy.inverse_transform([[scaled_price]])[0][0]
    return scaled_price

print(predict(2000,2,w,b))

def stochastic_gradient_descent(X,y_true,epochs,learning_rate = 0.001):
    import random 
    number_of_features = X.shape[1] # Number of columns 
    w = np.ones(shape=(number_of_features)) # it will assign 1 to each waits w1,w2 
    b = 0 # bias 
    total_samples = X.shape[0] # number of rows 
    
    cost_list = []
    epochs_list = [] 
    
    for i in range(epochs):
        random_index = random.randint(0,total_samples - 1)
        sample_X = X[random_index]
        sample_y = y_true[random_index]
        y_predicted = np.dot(w,sample_X.T) + b
        
        w_grad = -(2/total_samples) * (sample_X.T.dot(sample_y - y_predicted))
        b_grad = -(2/total_samples) * np.sum(sample_y - y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        cost = np.mean(np.square(sample_y - y_predicted))
        
        if i%10 == 0 :
            cost_list.append(cost)
            epochs_list.append(i)
        
           # print(f'waits:{w},bias:{b},cost:{cost},cost list:{cost_list},epochs:{epochs_list}') 

    return w,b,cost,cost_list,epochs_list

w_sgd,b_sgd,cost_sgd,cost_list_sgd,epochs_list_sgd = stochastic_gradient_descent(X_scaled,y_scaled.reshape(y_scaled.shape[0],),10000)
print('Stochcastic gradient descent waits is:\n',w_sgd)
print('Stochcastic gradient descent bias  is:\n',b_sgd)
print('Stochcastic gradient descent cost  is:\n',cost_sgd)

plt.xlabel('Stochcastic gradient epoch')
plt.ylabel('Stochcastic gradient cost')
plt.plot(epochs_list_sgd,cost_list_sgd)
plt.show()

print('Prediction using SGD Is:\n',predict(1200,2,w_sgd,b_sgd))
print('Prediction using Batch Is:\n',predict(1200,2,w,b))

def mini_batch_gradient_descent(X,y_true,epochs,batch,learning_rate = 0.001):
    import random 
    number_of_features = X.shape[1] # Number of columns 
    w = np.ones(shape=(number_of_features)) # it will assign 1 to each waits w1,w2 
    b = 0 # bias 
    total_samples = X.shape[0] # number of rows 
    
    cost_list = []
    epochs_list = [] 
    
    for i in range(epochs):
        random_index = random.sample(range(0,total_samples - 1),batch)
        sample_X = X[random_index]
        sample_y = y_true[random_index]
        y_predicted = np.dot(w,sample_X.T) + b
        
        w_grad = -(2/total_samples) * (sample_X.T.dot(sample_y - y_predicted))
        b_grad = -(2/total_samples) * np.sum(sample_y - y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        cost = np.mean(np.square(sample_y - y_predicted))
        
        if i%10 == 0 :
            cost_list.append(cost)
            epochs_list.append(i)
        
           # print(f'waits:{w},bias:{b},cost:{cost},cost list:{cost_list},epochs:{epochs_list}') 

    return w,b,cost,cost_list,epochs_list


w_batch,b_batch,cost_batch,cost_list_batch,epochs_list_batch = mini_batch_gradient_descent(X_scaled,y_scaled.reshape(y_scaled.shape[0],),10000,10)

print('Mini Batch gradient descent waits is:\n',w_batch)
print('Mini Batch gradient descent bias  is:\n',b_batch)
print('Mini Batch gradient descent cost  is:\n',cost_batch)

plt.xlabel('Mini Batch gradient epoch')
plt.ylabel('Mini Batch gradient cost')
plt.plot(epochs_list_batch,cost_list_batch)
plt.show()

print('Prediction using Mini Batch Is:\n',predict(1100,2,w_batch,b_batch))
print('Prediction using Batch Is:\n',predict(1100,2,w,b))