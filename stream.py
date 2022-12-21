#importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import Grouper
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing
from pandas import read_csv
from pylab import rcParams
from scipy.special import boxcox, inv_boxcox
from scipy import stats
import calmap
import calendar

def mean_absolute_percentage_error(test, predictions): 
    test, predictions = np.array(test), np.array(predictions)
    return np.mean(np.abs((test - predictions) / test)) * 100

#setting the page fevicon details
#st.set_page_config(page_title ="Gold Price Forecast Project")

#Title and other text details
st.title('Gold Price Prediction')


#Drop down menu - Select no.of days for prediction
df = pd.DataFrame({
    'first column': [30]
    })
nobs = st.selectbox(
    'Number of Days to Predict the Gold Price:',
     df['first column'])
#'You selected: ', nobsStrea

#File uploader
#uploaded_file = st.file_uploader("data.csv")
#if uploaded_file is not None:
  #series = pd.read_csv(uploaded_file)
  #st.write(series)
series = pd.read_csv("data.csv")

if st.button('Show Dataset'):
    st.header('Gold dataset')
    st.write(series)

st.write('---')

#Model pre-processing
series_2 = series.copy()
series = series.set_index(['date'])
series = series.astype(float)
#st.write(series)

#series.columns=["price"]
#series.index=pd.to_datetime(series.index)
#st.write(series)

dataframe = DataFrame(series.values)
dataframe.columns = ['price']
dataframe['price'] = stats.boxcox(dataframe['price'], lmbda=0.0)
#st.write(dataframe)

Train=dataframe.iloc[:1745,:]
Test=dataframe.iloc[1745:,]

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["price"],seasonal="add",trend="add",seasonal_periods=365).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
mean_absolute_percentage_error(pred_hwe_add_add,Test.price)

hwe_model_add_add = ExponentialSmoothing(dataframe["price"],seasonal="add",trend="add",seasonal_periods=365).fit()

Forecasted_price_5 = hwe_model_add_add.forecast(5)
Forecasted_price_10 = hwe_model_add_add.forecast(10)
Forecasted_price_15 = hwe_model_add_add.forecast(15)
Forecasted_price_20 = hwe_model_add_add.forecast(20)
Forecasted_price_25 = hwe_model_add_add.forecast(25)
Forecasted_price_30 = hwe_model_add_add.forecast(30)

Forecasted_price_true_5 = inv_boxcox(Forecasted_price_5, 0.0)
Forecasted_price_true_10 = inv_boxcox(Forecasted_price_10, 0.0)
Forecasted_price_true_15 = inv_boxcox(Forecasted_price_15, 0.0)
Forecasted_price_true_20 = inv_boxcox(Forecasted_price_20, 0.0)
Forecasted_price_true_25 = inv_boxcox(Forecasted_price_25, 0.0)
Forecasted_price_true_30 = inv_boxcox(Forecasted_price_30, 0.0)

dataframe_forecasted = DataFrame(Forecasted_price_true_30.values)
dataframe_forecasted.columns = ['Next 30 Days Price:']
st.write(dataframe_forecasted)

series_no_inx = series_2.reset_index()
series_no_inx_dec = series_no_inx[2050:]

series_no_inx_dec['price'].plot(figsize=(12,8),legend=True,label='Current Price [Showing only last 132 days for better view]')
Forecasted_price_true_5.plot(legend=True,label='Forecasted Price Price for 30-days')
plt.title('Gold Price Prediction');

calmap.calendarplot(df['price'], fillcolor='black',fig_kws=dict(figsize=(20, 15)))

#st.line_chart(data=Forecasted_price_true_30, width=0, height=0, use_container_width=True)
fig,ax = plt.subplots(figsize=(20,8))
plt.plot(np.arange(132), series_no_inx_dec['price'].values)
plt.plot(np.arange(132, 132+nobs), Forecasted_price_true_30.values)
plt.show()
st.pyplot(fig)
