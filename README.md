## Background:


A portfolio investment company makes investments in emerging markets around the world. Their goal is to establish a robust intelligent system to aid their value investing efforts using stock market data. They make investment decisions based on intrinsic value of companies and do not trade on the basis of daily market volatility.


## Data Description:

I am given a set of portfolio companies, trading data from emerging markets, from 2020 to 2021 Q1 stock prices. Each company stock is provided in different sheets. Each market's operating days vary based on the country of the company and the market the stocks are exchanged. I need to use only 2020 data and predict 2021 Q1 data.

Goal(s):
Predict stock price valuations on a daily, weekly and monthly basis. Recommend BUY, HOLD, SELL decisions. Maximize capital returns, minimize losses. Evaluate on the basis of capital returns. Use Bollinger Bands to measure your systems effectiveness.

## Methodology:

In this project, I used several algorithms to forecast, evaluate, & recommend buy/sell decisions. Below is the list of all techniques implemented:
1) Simple moving average (SMA)
2) Exponential moving average (EMA)
3) LSTM
4) Facebook Prophet

SMA, & EMA were used to establish baseline model for our more sophisticated algorithms such as LSTM and Prophet. For evaluation, Mean absolute percentage error (MAPE), equity curve, & backtesting strategy were used to used to assess our forecast accuracy where a good generalized model would return a lower MAPE and a high equity percentage. 

For my backtesting strategy, I made a one-time investment of 100,000 and measured its final return in terms of absolute value that is earnings and return on investment. It is to be noted that Bollinger Band strategy is one of the many strategies. There are many sophisticated strategies that enable higher frequency trading better suited for minimizing risks and optimizing profit. Given the size of the actual dataset (2 years worth data), I opted for a simpler trading strategy.

The goal here was to evaluate forecasting methods such as LSTM v/s Prohphet and use backtesting strategy as a litmus test to evaluate forecasting performance. MAPE was used to measure the accuracy of the models with respect to actual prices. 

When considering a foercasting model, an ideal model should have lower MAPE and higher ROI. lower MAPE results, higher trade frequencies/ROI are evident when using LSTM compared to Prophet.


## Conclusion:

My focus algorithm for this project was LSTM v/s Prophet. My analysis showed LSTM was a much better predictor compared to the black box-Prophet model. 3 metrics were used to evaluate forecasting using trading strategy namely MAPE, equity curve, & bollinger band strategy. 


