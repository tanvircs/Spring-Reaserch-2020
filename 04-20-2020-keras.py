#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from keras.layers import LSTM
from keras.models import load_model
from keras.regularizers import l1,l2


# In[2]:


df = pd.read_csv('processed_data.csv',index_col='TIME',parse_dates=True)
df.index.freq = 'D'
df.head()


# In[3]:


df.plot(figsize=(12,8))


# In[4]:


results = seasonal_decompose(df['CONG'])
results.observed.plot(figsize=(12,4))


# In[5]:


results.trend.plot(figsize=(12,4))


# In[6]:


results.seasonal.plot(figsize=(16,4))


# In[7]:


results.resid.plot(figsize=(12,4))


# In[8]:


len(df)


# In[45]:


train = df.iloc[:755]
test = df.iloc[755:]


# In[46]:


scaler = MinMaxScaler()


# In[47]:


scaler.fit(train)


# In[48]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[49]:


# scaled_train


# In[50]:


n_input = 7
n_features = 1
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[51]:


scaled_train[:10]


# In[52]:


train_generator[0]


# In[53]:


X, y = train_generator[0]


# In[54]:


X.shape


# In[55]:


y.shape


# In[56]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])


# In[57]:


model.summary()


# In[58]:


earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


# In[59]:


history = model.fit_generator(train_generator, callbacks = [earlystopper], epochs=12)


# In[24]:


# model.fit_generator(train_generator, epochs=3)


# In[60]:


model.history.history.keys()


# In[82]:


from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = (14,5)
myloss = model.history.history['loss']
plt.plot(range(len(myloss)), myloss)
plt.tight_layout()
plt.title('Training Loss', fontsize=16)
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.savefig('LSTM Loss', dpi=400, bbox_inches = 'tight')


# In[62]:


first_eval_batch = scaled_train[-7:]


# In[63]:


# first_eval_batch


# In[64]:


first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))


# In[65]:


first_eval_batch


# In[66]:


model.predict(first_eval_batch)


# In[67]:


test_predictopn = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape(1,n_input,n_features)

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictopn.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)


# In[68]:


# test_predictopn


# In[69]:


true_prediction = scaler.inverse_transform(test_predictopn)


# In[70]:


# true_prediction


# In[71]:


test['Predictions'] = true_prediction


# In[72]:


test.head()


# In[73]:


test.plot(figsize=(12,5))


# In[81]:


plt.figure(figsize=(20, 5))
plt.plot(test.index, test['CONG'])
plt.plot(test.index, test['Predictions'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()


# In[75]:


print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(test['CONG'], test['Predictions'])))


# In[ ]:





# In[ ]:





# In[ ]:





# In[83]:


model.save('lstm_daily.h5')


# In[42]:


# pwd


# In[43]:


# from keras.models import load_model


# In[44]:


# new_model = load_model('lstm_daily.h5')


# In[ ]:




