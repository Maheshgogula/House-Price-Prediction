#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


# Training data
data = pd.read_excel("using machine learning(1).xlsx")
data.head()


# In[3]:


data.columns


# In[4]:


data.info()


# # EDA

# In[5]:


# Function to plot a boxplot
def box(data):
    plt.figure(figsize=(10, 5))
    plt.boxplot(data)
    plt.show()


# In[6]:


# Boxplot to check outliers form SalePrice variable
box(data.SalePrice)


# In[7]:


# Boxplot of LotArea variable
box(data.LotArea)


# <h4>Function to remove Outliers

# In[8]:


# Function to remove outliers
def drop_outiers(data, field):
    irq = 1.5 * (np.percentile(data[field], 75) - np.percentile(data[field], 25))
    data1 = data[data[field] > (irq + np.percentile(data[field], 75))]
    data2 = data[data[field] < (np.percentile(data[field], 25) - irq)]
    return data1, data2


# In[9]:


data1_up = drop_outiers(data, 'SalePrice')[0]
print('Outliers size for SalePrice', data1_up.shape[0])
# Dropping the outliers
data.drop(data1_up.index, inplace=True)


# In[10]:


# Dropping the outliers
data2_up = drop_outiers(data, 'LotArea')[0]
data2_down = drop_outiers(data, 'LotArea')[1]
print('Outliers size for LotArea', data2_up.shape[0] + data2_down.shape[0])
data.drop(data2_up.index, inplace=True)
data.drop(data2_down.index, inplace=True)

# New Sape of the data after removing outliers
print('New shape of the data after removing outliers in', data.shape)


# <h4>No missing values

# Distribution between variables

# In[11]:


sns.pairplot(kind='scatter',diag_kind='kde',data=data)


# We can see that SalePrice has nomal distribution with long rigt tail as well as LotArea
# 
# MoSold also has normal distribution and YrSold has Multi-gaussian distribution because it has integrated all the MoSold.

# <h4>correlation

# In[12]:


plt.figure(figsize=(10,6))
sns.heatmap(data=data.corr(), center=True,annot=True, cmap='ocean_r')
plt.show()


# It has been shown that there is no correlation between the input vairables,
# therefore we can use linear regression for the SalePrice prediction
# 
# It also cleary showing that YrSold(Year Sold) is having negative correlation
# with other variables.Hence he can conclude that it can be useless in our prediction

# <h4>Applying Linear Regression to built a model for prediction

# In[13]:


# Standarization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns = ['MoSold', 'YrSold', 'LotArea', 'BedroomAbvGr', 'SalePrice']
data = pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns)
predictors = ['MoSold', 'YrSold', 'LotArea', 'BedroomAbvGr']
X = data[predictors] # Predictors

#X = scaler.fit_transform(X)
y = data.iloc[:,-1] # Target variable
#y = scaler.fit_transform(np.array(y).reshape(-1, 1))


# In[14]:


data.head()


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = data.copy()
def scaler(data):
    scaler = StandardScaler()
    new_data = scaler.fit_transform(data)
    return new_data

df = scaler(df)


# In[16]:


df = pd.DataFrame(df, columns=columns )
predictors = ['MoSold', 'YrSold', 'LotArea', 'BedroomAbvGr']
X = data[predictors] # Predictors
y = data.iloc[:,-1]


# In[17]:


# Splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[18]:


# Instantiate the model and fit it
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[19]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[20]:


# calculating the r2 score value
from sklearn.metrics import r2_score
print(r2_score(np.array(y_test).reshape(-1, 1), pred_test) * 100)


# In[21]:


from termcolor import colored as cl
from sklearn.metrics import explained_variance_score as evs
print(cl('Explained of linear model is {}'.format(evs(y_test,pred_test)), attrs = ['bold']))


# <h4>Applying KNN algorithm to built a model for prediction

# In[22]:


from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=2)
regressor.fit(data[predictors], data.SalePrice)


# In[23]:


from sklearn.metrics import mean_squared_error
import math

pre = regressor.predict(X_test)
pre_y = regressor.predict(X_train)

# print('Training Error',math.sqrt(mean_squared_error(y_train, pre_y)))
# print('Testing Error',math.sqrt(mean_squared_error(y_test, pre)))

#print(r2_score(np.array(y_test).reshape(-1, 1), pre) * 100)


# In[24]:


print(r2_score(np.array(y_test).reshape(-1, 1), pre) * 100)


# In[25]:


print(cl('Explained of linear model is {}'.format(evs(y_test,pre)), attrs = ['bold']))


# In[26]:


error = []

from sklearn.metrics import mean_squared_error

# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_squared_error(y_test, pred_i)
    error.append(mae)


# In[27]:


import matplotlib.pyplot as plt 

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xticks([i for i in range(0, 40)])
         
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')


# In[28]:


from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(data[predictors], data.SalePrice)


# In[29]:


pre = regressor.predict(X_test)
pre_y = regressor.predict(X_train)


# In[30]:


print(r2_score(np.array(y_test).reshape(-1, 1), pre) * 100)


# In[31]:


print(cl('Explained of linear model is {}'.format(evs(y_test,pre)), attrs = ['bold']))


# In[32]:


import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError


# In[33]:


# def scale_datasets(x_train, x_test):

#     """
#     Standard Scale test and train data
#     Z - Score normalization
#     """
#     standard_scaler = StandardScaler()
#     x_train_scaled = pd.DataFrame(
#       standard_scaler.fit_transform(x_train),
#       columns=predictors
#     )
#     x_test_scaled = pd.DataFrame(
#       standard_scaler.transform(x_test),
#       columns = predictors
#     )
#     return x_train_scaled, x_test_scaled
# x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)


# In[34]:


hidden_units1 = 120
hidden_units2 = 160
hidden_units3 = 130
learning_rate = 0.001
# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
    model = Sequential([
    Dense(hidden_units1, activation='relu'),
    Dense(hidden_units2, activation='relu'),
    Dense(hidden_units3, activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
  ])
    return model
# build the model
model = build_model_using_sequential()


# In[35]:


# # Splitting data into train and test data
# X = data[predictors].values
# y = data.iloc[:,-1].values

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[36]:


# loss function
msle = MeanSquaredLogarithmicError()
model.compile(
    loss=msle, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[msle]
)
# train the model
history = model.fit(
    X_train,
    y_train, 
    epochs=5, 
    batch_size=64,
    validation_split=0.2
)


# In[37]:


y_test


# In[38]:


b = model.predict(X_test)
v = b.reshape(-1,)


# In[39]:


r2_score(y_test, v)*100


# In[40]:


def plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  plt.show()
# Plot the history
plot_history(history, 'mean_squared_logarithmic_error')


# In[41]:


import matplotlib.pyplot as plt 
import pandas as pd 
r1=pd.DataFrame(history.history) 
r1['Epcohs'] = history.epoch 
print(r1.tail()) 
plt.plot(r1['Epcohs'],r1['loss'],label='Train_loss') 
plt.plot(r1['Epcohs'],r1['val_loss'],label='Test_loss')
plt.xlabel('Epcohs') 
plt.ylabel('Loss') 
plt.legend() 
plt.show() 
 


# In[42]:


plt.plot(r1['Epcohs'],r1['mean_squared_logarithmic_error'],label='Train_accuracy') 
plt.plot(r1['Epcohs'],r1['val_mean_squared_logarithmic_error'],label='Test_accuracy')
plt.xlabel('Epcohs') 
plt.ylabel('accuracy') 
plt.legend() 
plt.show()


# In[ ]:




