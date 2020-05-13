#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as smt
import itertools

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Sequential
pd.pandas.set_option('display.max_columns', None)
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols


# In[2]:


df = pd.read_csv('2018only.csv', parse_dates=True, index_col='TIME')


# In[3]:


df.head()


# In[4]:


plt.rcParams["figure.figsize"]=(16,6)
plt.plot(df.iloc[:235])


# In[5]:


df1 = df.iloc[:235]


# In[6]:


df1


# In[7]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[8]:


adf_test(df1)


# In[9]:


ts_log = np.log(df1)
plt.plot(ts_log)


# In[10]:


moving_avg = ts_log.rolling(30).mean()


# In[11]:


plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[12]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(30)


# In[13]:


ts_log_moving_avg_diff.dropna(inplace=True)
adf_test(ts_log_moving_avg_diff)


# In[14]:


def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=30).mean()
    rolling_std = timeseries.rolling(window=30).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[15]:


get_stationarity(ts_log_moving_avg_diff)


# In[16]:


expwighted_avg = ts_log.ewm(span=7).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


# In[17]:


ts_log_ewma_diff = ts_log - expwighted_avg
adf_test(ts_log_ewma_diff)
get_stationarity(ts_log_ewma_diff)


# In[18]:


auto_arima(ts_log_moving_avg_diff,seasonal=True,m=30).summary()


# In[19]:


len(ts_log_moving_avg_diff)


# In[20]:


train = ts_log_moving_avg_diff.iloc[:180]
test = ts_log_moving_avg_diff.iloc[180:]


# In[37]:


model = SARIMAX(train, order=(2,0,2), seasonal_order=(1,0,1,30),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[38]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(2,1,2)(2,1,2,30) Predictions')


# In[39]:


title='Traffic Congestion'
ylabel='Level of Congestion'
xlabel=''

ax = test.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[40]:


df_new = pd.read_csv('cleaned---monthly_Without_aggregation.csv', index_col='TIME', parse_dates=True)


# In[76]:


import pandas as pd
import pandas as T
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import r2_score
from keras.optimizers import Adam


# In[42]:


len(df_new)


# In[47]:


train = df_new.iloc[:240]
test = df_new.iloc[240:]


# In[155]:


scaler = MinMaxScaler()
scaler.fit(train)


# In[156]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[157]:


n_input = 30
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=30)


# In[158]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[159]:


model.summary()


# In[160]:


model.fit_generator(generator, epochs=30)


# In[161]:


model.history.history.keys()


# In[162]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[163]:


first_eval_batch = scaled_train[-30:]


# In[164]:


first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))


# In[165]:


model.predict(first_eval_batch)


# In[166]:


scaled_test[0]


# In[167]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))


# In[168]:


np.append(current_batch[:,1:,:],[[[99]]],axis=1)


# In[169]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[170]:


test_predictions


# In[171]:


scaled_test


# In[172]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[173]:


true_predictions


# In[174]:


test['Predictions'] = true_predictions


# In[175]:


test.plot(figsize=(12,8))


# In[176]:


test


# In[177]:


len(ts_log_moving_avg_diff)


# In[ ]:





# In[ ]:




