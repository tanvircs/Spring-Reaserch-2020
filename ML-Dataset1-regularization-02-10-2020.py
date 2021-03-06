#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np
import tensorflow.contrib.keras as keras
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import r2_score


# In[3]:


read_file = pd.read_excel (r'C:/Users/T_Tan/OneDrive/Desktop/arups_data/selected-features-1-c1c4c6.xlsx')
read_file.to_csv (r'dataset1.csv', index = None, header=True)


# In[4]:


dataset1= pd.read_csv("dataset1.csv", 
                header=None, names=["f1","f2","f3","f4","f5","f6","f7","f8","label"])


# In[5]:


train = dataset1.loc[0:629]
Xtrain = train.iloc[:,0:8]
Ytrain = train.iloc[:,8]
test = dataset1.loc[630:944]
Xtest = test.iloc[:,0:8]
Ytest = test.iloc[:,8]


# In[6]:


scaler = StandardScaler()

X_train = scaler.fit_transform(Xtrain)
y_train = Ytrain.values.reshape(-1,1)
y_train = scaler.fit_transform(y_train)

X_test = scaler.fit_transform(Xtest)
y_test = Ytest.values.reshape(-1,1)
y_test = scaler.fit_transform(y_test)


# In[7]:


model = Sequential()
# model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='elu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='elu'))
# model.add(Dropout(0.2))
model.add(Dense(4, activation='elu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1,))
model.compile(Adam(lr=0.003), loss='mse', metrics=['mae','mse'])


# In[8]:


earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


# In[9]:


history = model.fit(X_train, y_train, epochs = 60, batch_size=200, validation_split = 0.2, shuffle = True, verbose = 1, 
                    callbacks = [earlystopper])


# In[10]:


y_test_pred = model.predict(X_test)


# In[11]:


print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))


# In[12]:


plt.plot(y_test,'b',label='Actual')
plt.plot(y_test_pred,'g',label='Predicted')
plt.legend(fontsize=14)
plt.tight_layout()
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('Actual Data', fontsize=14)
plt.ylabel('Predicted Data', fontsize=14)
plt.savefig('Prediction1Regularization', dpi=400, bbox_inches = 'tight')


# In[13]:


pred = scaler.inverse_transform(y_test_pred)


# In[14]:


prediction = pd.DataFrame(pred, columns=['predictions']).to_csv('prediction1Regularization.csv')


# In[15]:


model.evaluate(X_test,y_test)

