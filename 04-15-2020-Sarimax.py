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

from sklearn.model_selection import train_test_split
from keras.models import Sequential
pd.pandas.set_option('display.max_columns', None)
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols


# In[2]:


df = pd.read_csv('Chicago_Traffic_Tracker_Monthly.csv', parse_dates=True)
df.head()


# In[3]:


u_Region = df['REGION'].unique()
u_Region


# In[4]:


df_Austin = df[df['REGION']=='Austin']
df_Austin.head()


# In[5]:


df_Austin.shape


# In[6]:


df_Austin['SPEED'].max()


# In[7]:


df_Austin['TIME'].max()


# In[8]:


df_Austin['TIME'].min()


# In[9]:


df_Austin=df_Austin.sort_values(by='TIME')
df_Austin.head()


# In[10]:


df_Austin.shape


# In[11]:


df.shape


# In[12]:


df_Austin.describe()


# In[13]:


df_Austin['SPEED'].isnull().count()


# In[14]:


df_Austin['TIME'] = df_Austin['TIME'].astype('datetime64[ns]')


# In[15]:


df_Austin=df_Austin.sort_values(by='TIME')


# In[16]:


df_Austin.head()


# In[17]:


df_Austin.tail()


# In[18]:


start= pd.Timestamp("2018-03-09 15:40:41")
end = pd.Timestamp("2020-4-10 14-31-35")


# In[19]:


df_Austin_con = df_Austin[['TIME','SPEED','BUS_COUNT','NUM_READS']]


# In[20]:


df_Austin_con.shape


# In[21]:


# df_Austin_con['CONG']= df_Austin_con[df_Austin_con['SPEED']/40]
df_Austin_con['CONG'] = df_Austin_con['SPEED'].map(lambda x: float(x/40))


# In[22]:


df_Austin_con.head()


# In[23]:


df_Austin_con.shape


# In[24]:


df_Aus_Cong= df_Austin_con[['TIME','CONG']]
df_Aus_Cong.shape


# In[25]:


df_Aus_Cong.head(5)


# In[ ]:





# In[26]:


df_Aus_Cong = df_Aus_Cong[(df_Aus_Cong['TIME']>='2018-03-10 00:10:28') & (df_Aus_Cong['TIME']<'2020-04-10  00:01:29') ]


# In[27]:


df_Aus_Cong.head(5)


# In[28]:


df_Aus_Cong.tail(6)


# In[29]:


df_Aus_Cong.set_index('TIME',inplace=True)


# In[30]:


df_Aus_Cong.head()


# In[31]:


output = df_Aus_Cong.resample('D').mean()
output.head()


# In[32]:


day = df_Austin_con[(df_Austin_con['TIME']>='2018-03-10 00:10:28') & (df_Austin_con['TIME']<='2018-03-11 00:10:20') ]
day.shape


# In[33]:


day.head()


# In[34]:


day.tail()


# In[35]:


s=day['CONG'].sum()


# In[36]:


s/114


# In[37]:


output.shape


# In[38]:


# df_nan= pd.DataFrame(output).set_index('CONG')


# In[39]:


df1 = output[output.isna().any(axis=1)]


# In[40]:


df1.shape


# In[41]:


df1


# In[42]:


output.head(30)


# In[43]:


[output['CONG'].fillna(output['CONG'].mean(), inplace=True) for col in output.columns]


# In[ ]:





# In[44]:


df1 = output[output.isna().any(axis=1)]
df1.head(15)


# In[45]:


output.to_csv('cleaned---monthly_Without_aggregation.csv')


# In[98]:


len(output)


# In[293]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
train = output.iloc[:680]
test = output.iloc[680:]

fitted_model = ExponentialSmoothing(train['CONG'], trend='mul',seasonal= 'mul',seasonal_periods=120).fit()


# In[294]:


test_predictions = fitted_model.forecast(80)


# In[295]:


test_predictions


# In[296]:


train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')
test_predictions.plot(legend=True,label='Prediction')


# In[297]:


train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')


# In[298]:


train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')
test_predictions.plot(legend=True,label='Prediction',xlim=['2019-08-10 17:30:00','2021-04-15 17:30:00'])


# In[284]:





# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# train = df_log_exp_decay.iloc[:600]
# test = df_log_exp_decay.iloc[600:]

# fitted_model = ExponentialSmoothing(train['CONG'], trend='mul',seasonal= 'mul',seasonal_periods=182).fit()
# test_predictions = fitted_model.forecast(90)

# train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
# test['CONG'].plot(legend=True,label='Test')
# test_predictions.plot(legend=True,label='Prediction')


# train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
# test['CONG'].plot(legend=True,label='Test')



# train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
# test['CONG'].plot(legend=True,label='Test')
# test_predictions.plot(legend=True,label='Prediction',xlim=['2020-02-25 17:30:00','2020-04-15 17:30:00'])


# In[53]:


result = seasonal_decompose(output.CONG.values, freq=180)  # model='mul' also works
result.plot();


# In[54]:


from pandas.plotting import lag_plot


# In[55]:


lag_plot(output['CONG']);


# In[56]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[57]:


acf(output['CONG'])


# In[94]:


title = 'Autocorrelation: Traffic Congestion'
lags = 90
plot_acf(output,title=title,lags=lags);


# In[92]:


title='Partial Autocorrelation: Traffic Congestion'
lags=90
plot_pacf(df_log_exp_decay,title=title,lags=lags);


# In[95]:


model = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,14),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[96]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(2,0,2)(2,1,2,30) Predictions')


# In[97]:


title='Traffic Congestion'
ylabel='Level of Congestion'
xlabel=''

ax = test.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


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


# In[61]:


from pylab import rcParams
rcParams['figure.figsize'] = 14,7


# In[62]:


adf_test(output)


# In[63]:


df_log = np.log(output)
plt.plot(df_log)


# In[64]:


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


# In[65]:


rolling_mean = df_log.rolling(window=30).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
get_stationarity(df_log_minus_mean)


# In[66]:


adf_test(df_log_minus_mean)


# In[67]:


rolling_mean_exp_decay = df_log.ewm(span=30, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)


# In[68]:


adf_test(df_log_exp_decay)


# In[69]:


df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)


# In[70]:


adf_test(df_log_shift)


# In[71]:


auto_arima(df_log_exp_decay,seasonal=True,m=30).summary()


# In[72]:


# p = d = q = range(0, 2)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]


# In[73]:


# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = SARIMAX(df_log_exp_decay,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}30 - AIC:{}'.format(param,param_seasonal,results.aic))
#         except: 
#             continue


# In[74]:


# print ll


# In[ ]:





# In[75]:


len(df_log_exp_decay)


# In[76]:


train = df_log_exp_decay.iloc[:600]
test = df_log_exp_decay.iloc[600:]


# In[77]:


model = SARIMAX(train, order=(2,1,2), seasonal_order=(2,1,2,30),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[78]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(2,0,2)(2,1,2,30) Predictions')


# In[79]:


title='Traffic Congestion'
ylabel='Level of Congestion'
xlabel=''

ax = test.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model = SARIMAX(train, order=(2,0,2), seasonal_order=(1,0,[1],7),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[ ]:





# In[ ]:





# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(2,0,2)(1,0,[1],7) Predictions')


# In[ ]:


title='Traffic Congestion'
ylabel='Level of Congestion'
xlabel=''

ax = test.plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
train = df_log_exp_decay.iloc[:600]
test = df_log_exp_decay.iloc[600:]

fitted_model = ExponentialSmoothing(train['CONG'], trend='add',seasonal= 'mul',seasonal_periods=182).fit()
test_predictions = fitted_model.forecast(90)

train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')
test_predictions.plot(legend=True,label='Prediction')


train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')



train['CONG'].plot(legend=True,label='Train',figsize=(15,8))
test['CONG'].plot(legend=True,label='Test')
test_predictions.plot(legend=True,label='Prediction',xlim=['2020-02-25 17:30:00','2020-04-15 17:30:00'])


# In[ ]:


train.shape


# In[ ]:


df_train_neg= train[train['CONG']<0]
df_train_neg.shape


# In[ ]:


df_test_neg = test[test['CONG']<0]
df_test_neg.shape


# In[ ]:





# In[ ]:


df_neg = df_log_exp_decay[df_log_exp_decay['CONG']<0]
df_neg.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




