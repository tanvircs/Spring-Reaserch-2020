#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tools.eval_measures import rmse, mse
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[2]:


df = pd.read_csv('cleaned---monthly_Without_aggregation.csv')
df.head()


# In[3]:


# df[] = df.astype(np.int64)


# In[4]:


df.info()


# In[5]:


df.columns = ['ds','y']


# In[6]:


df.head()


# In[7]:


df['ds'] = pd.to_datetime(df['ds'])


# In[8]:


df.head()


# In[9]:


m = Prophet()
m.fit(df)


# In[10]:


future = m.make_future_dataframe(periods=90, freq='D')


# In[11]:


# future.tail()


# In[12]:


df.tail()


# In[13]:


len(df)


# In[14]:


# len(future)


# In[15]:


forecast = m.predict(future)


# In[16]:


# forecast.head()


# In[17]:


# forecast.columns


# In[18]:


forecast[['ds','yhat_lower','yhat_upper','yhat']].tail(30)


# In[19]:


m.plot(forecast);


# In[20]:


forecast.plot(x='ds',y='yhat',figsize=(14,5))


# In[21]:


# m.plot_components(forecast);


# In[22]:


len(df)


# In[23]:


train = df.iloc[:267]
test = df.iloc[267:]


# In[24]:


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=30, freq='D')
forecast = m.predict(future)


# In[25]:


ax = forecast.plot(x='ds', y='yhat', label='Prediction', legend=True, figsize=(15,5))
test.plot(x='ds',y='y', label='True Test Data', legend=True, ax=ax, xlim=('2018-12-01','2018-12-30'))


# In[26]:


test


# In[27]:


predictions = forecast.iloc[-30:]['yhat']


# In[28]:


predictions


# In[29]:


test['y']


# In[30]:


rmse(predictions,test['y'])


# In[31]:


test.mean()


# In[32]:


m.plot_components(forecast);


# In[33]:


train.head(30)


# In[34]:


initial = 5*30
initial = str(initial)+' days'


# In[35]:


initial


# In[36]:


period = 5*30
period = str(period)+' days'


# In[37]:


period


# In[38]:


horizon = 30
horizon = str(horizon)+' days'


# In[39]:


horizon


# In[40]:


df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)


# In[41]:


df_cv.head(30)


# In[42]:


len(df_cv)


# In[43]:


plot_cross_validation_metric(df_cv, metric='rmse');


# In[44]:


plot_cross_validation_metric(df_cv, metric='mape');


# In[45]:


from fbprophet.plot import add_changepoints_to_plot


# In[46]:


fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)


# In[47]:


m = Prophet(seasonality_mode='additive', growth='linear', daily_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(30, freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)


# In[48]:


m.plot_components(forecast);


# In[49]:


m = Prophet(seasonality_mode='multiplicative', daily_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(30, freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)


# In[50]:


m.plot_components(forecast);


# # Hourly Data

# In[51]:


import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tools.eval_measures import rmse
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[52]:


df1 = pd.read_csv('processed_data_hourly.csv')
df1.head()


# In[53]:


del df1['Unnamed: 0']


# In[54]:


df1.head()


# In[55]:


df1.columns = ['ds','y']
df1['ds'] = pd.to_datetime(df1['ds'])
df1.head()


# In[ ]:





# In[56]:


len(df1)


# In[57]:


train = df1.iloc[:82000]
test = df1.iloc[82000:]


# In[58]:


m = Prophet(seasonality_mode='additive', daily_seasonality=True, yearly_seasonality=True)
m.fit(train)
future = m.make_future_dataframe(periods=19058, freq='H')
forecast = m.predict(future)


# In[59]:


test.tail()


# In[60]:


ax = forecast.plot(x='ds', y='yhat', label='Prediction', legend=True, figsize=(15,5))
test.plot(x='ds',y='y', label='True Test Data', legend=True, ax=ax, xlim=('2020-04-04 00:10:00','2020-04-09 00:10:00'))


# In[61]:


predictions = forecast.iloc[-19058:]['yhat']


# In[62]:


mse(predictions,test['y'])


# In[63]:


m.plot_components(forecast);


# In[64]:


fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)


# In[65]:


m = Prophet(seasonality_mode='multiplicative', growth='linear', daily_seasonality=True)
m.fit(df1)
future = m.make_future_dataframe(5, freq='H')
forecast = m.predict(future)
fig = m.plot(forecast)
plt.title('Prophet Prediction Hourly', fontsize=16)
plt.savefig('Prophet Prediction Hourly', dpi=400, bbox_inches = 'tight');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




