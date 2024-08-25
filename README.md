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

<img width="897" alt="image" src="https://github.com/user-attachments/assets/2b5ded6d-2bae-421c-8595-12b87f7a3c15">
<img width="896" alt="image" src="https://github.com/user-attachments/assets/45bce9dd-664f-48fa-a0b8-05277c667cfd">


SMA, & EMA were used to establish baseline model for our more sophisticated algorithms such as LSTM and Prophet. For evaluation, Mean absolute percentage error (MAPE), & backtesting were used to assess our forecast accuracy where a good generalized model would return a lower MAPE and a high ROI percentage. 

There are numerous ways to develop trading strategies, but for my project, I used Bollinger-Band strategy to simulate buy/sell signals. The premise of how Bollinger-band strategy works is as follows:

•	We buy when the closing price is under the lower Bollinger Band
•	We sell when the price crosses above the upper Bollinger Band

To evaluate or quantify the results of my strategy, I performed a backtesting where, I made a one-time investment of 100,000 and measured its final return in terms of absolute value that is earnings and return on investment. It is to be noted that Bollinger Band strategy is one of the many strategies. There are many sophisticated strategies that enable high-frequency trading better suited for minimizing risks and optimizing profit. Given the size of the actual dataset (2 years worth data), I opted for a simpler trading strategy.

The goal here was to evaluate forecasting methods such as LSTM v/s Prohphet and use backtesting strategy as a litmus test to evaluate forecasting performance. MAPE was used to measure the accuracy of the models with respect to actual prices. 
 

<img width="890" alt="image" src="https://github.com/user-attachments/assets/1cdd4e37-525e-4e38-963e-a611eafe644c">


## Conclusion:

My focus algorithms for this project was LSTM v/s Prophet. My analysis showed LSTM was a much better predictor than the black box-Prophet model. MAPE was used to evaluate forecasting results and earnings/ROI was used to determine Bollinger band strategy's success. Here an ideal forecast would correspond to results with lower MAPE and higher trade frequencies/ROI. Below is an example of 'koc_holdings' stock comparing the final results of the forecasts between LSTM and Prophet. 

<img width="890" alt="image" src="https://github.com/user-attachments/assets/543a0502-4a02-4385-9626-e305d75c87f6">

From above it is quite apparent that the Prophet forecasted poorly thus affecting backtesting's success of returning suitable earnings/ROI. On the other hand, LSTM forecasted well with its trading strategy's results depicted below:

![image](https://github.com/user-attachments/assets/a8822139-38e5-4ed1-ad8d-956a53daff68)

Our ROI for 'koc_holding' was -4.76% which is not that bad considering the sudden dip in tail prices that were not observed in the first or the second month of forecasting. From here one should also infer that using ROI as a measure for success is relative. ROI should be used as a metric for success only if:

1) There is ample historical data to train on
2) Deployed trading strategy is highly sound such that minimal loss occurs

Thus due to the above reasons, I used MAPE as my main metric. There are many ways to develop robust trading strategies such as combining Bollinger band with RSI (Relative Strength Index), Using SMA to determine upper & lower bands instead of actual prices and many more but given our size of the dataset, a simple bollinger-band strategy was sufficient. 



