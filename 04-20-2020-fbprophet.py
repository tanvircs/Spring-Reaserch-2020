#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tools.eval_measures import rmse
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[2]:


df = pd.read_csv('processed_data.csv')
df.head()


# In[3]:


df.columns = ['ds','y']


# In[4]:


df['ds'] = pd.to_datetime(df['ds'])
df.head()


# In[5]:


len(df)


# In[6]:


train = df.iloc[:702]
test = df.iloc[702:]


# In[7]:


m = Prophet(seasonality_mode='additive', daily_seasonality=True, yearly_seasonality=True)
m.fit(train)
future = m.make_future_dataframe(periods=60, freq='H')
forecast = m.predict(future)


# In[8]:


ax = forecast.plot(x='ds', y='yhat', label='Prediction', legend=True, figsize=(15,5))
test.plot(x='ds',y='y', label='True Test Data', legend=True, ax=ax, xlim=('2020-02-10','2020-04-09'))


# In[9]:


predictions = forecast.iloc[-60:]['yhat']


# In[10]:


rmse(predictions,test['y'])


# In[11]:


m.plot_components(forecast);
plt.tight_layout()
# plt.title('Plot Components', fontsize=16)
# plt.xlabel('', fontsize=14)
# plt.ylabel('Level of Congestion', fontsize=14)
plt.savefig('Plot Components', dpi=400, bbox_inches = 'tight')


# In[12]:


initial = 5*30
initial = str(initial)+' days'

period = 5*30
period = str(period)+' days'

horizon = 30
horizon = str(horizon)+' days'

df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)


# In[13]:


# prophet_rmse_error = rmse(df['y'].iloc[-60:], df['yhat'])
# prophet_mse_error = prophet_rmse_error**2
# mean_value = df['y'].iloc[-60:].mean()

# print(f'MSE Error: {prophet_mse_error}\nRMSE Error: {prophet_rmse_error}\nMean: {mean_value}')


# In[14]:


plot_cross_validation_metric(df_cv, metric='rmse');
plt.title('RMSE', fontsize=16)
plt.savefig('Prophet RMSE', dpi=400, bbox_inches = 'tight');


# In[15]:


plot_cross_validation_metric(df_cv, metric='mape');
plt.title('MAPE', fontsize=16)
plt.savefig('Prophet MAPE', dpi=400, bbox_inches = 'tight');


# In[16]:


fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)
# plt.title('Trend Analysis', fontsize=16)
# plt.savefig('Prophet Trend', dpi=400, bbox_inches = 'tight');


# # Hourly Data

# In[18]:


m = Prophet(seasonality_mode='multiplicative', growth='linear', daily_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(30, freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)
plt.title('Prophet Prediction', fontsize=16)
plt.savefig('Prophet Prediction', dpi=400, bbox_inches = 'tight');


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




