# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

from sklearn.ensemble import RandomForestRegressor


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for training
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train['year'] = df_train['ds'].dt.year
df_train['month'] = df_train['ds'].dt.month
df_train['day'] = df_train['ds'].dt.day

X_train = df_train[['year', 'month', 'day']]
y_train = df_train['y']

# Train Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Make predictions
last_date = data.iloc[-1]['Date']
prediction_dates = pd.date_range(last_date, periods=period, freq='d')
df_predict = pd.DataFrame({'ds': prediction_dates})
df_predict['year'] = df_predict['ds'].dt.year
df_predict['month'] = df_predict['ds'].dt.month
df_predict['day'] = df_predict['ds'].dt.day

X_predict = df_predict[['year', 'month', 'day']]
y_predict = regressor.predict(X_predict)

forecast = pd.DataFrame({'ds': prediction_dates, 'yhat': y_predict})

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="forecast"))
fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
