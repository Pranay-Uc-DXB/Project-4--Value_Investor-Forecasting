## Background:


A portfolio investment company makes investments in emerging markets around the world. Their goal is to establish a robust intelligent system to aid their value investing efforts using stock market data. They make investment decisions based on intrinsic value of companies and do not trade on the basis of daily market volatility.


## Data Description:

I am given a set of portfolio companies, trading data from emerging markets, from 2020 to 2021 Q1 stock prices. Each company stock is provided in different sheets. Each market's operating days vary based on the country of the company and the market the stocks are exchanged. I need to use only 2020 data and predict 2021 Q1 data.

Goal(s):
Predict stock price valuations on a daily, weekly and monthly basis. Recommend BUY, HOLD, SELL decisions. Maximize capital returns, minimize losses. Ideally a loss should never happen. Evaluate on the basis of capital returns. Use Bollinger Bands to measure your systems effectiveness.

## Methodology:

In this project, I used several algorithms to forecast, evaluate, & recommend buy/sell decisions. Below is the list of all techniques implemented:
1) Simple moving average (SMA)
2) Exponential moving average (EMA)
3) ARIMA
4) LSTM
5) Facebook Prophet

SMA, EMA, & ARIMA were used to establish baseline model for our more sophisticated algorithms such as LSTM and Prophet. For evaluation, Mean absolute percentage error (MAPE) and equity curve returns were used to assess our forecast accuracy where a good generalized model would return a lower MAPE and a high equity percentage. 


## Conclusion:

My focus algorithm for this project was LSTM v/s Prophet. My analysis suggested that LSTM was a better predictor 
