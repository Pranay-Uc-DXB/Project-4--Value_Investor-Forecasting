# %%
# Importing libraries relevant to Moving Avg, Exp. Moving Avg, LSTM, and Facebook Prophet

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *

from sklearn.preprocessing import  MinMaxScaler
from statsmodels.tsa.api import SimpleExpSmoothing

from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# %%
stock_Data_path=pd.ExcelFile('../Data/2020Q1Q2Q3Q4-2021Q1.xlsx')

# %%
sheet_names=stock_Data_path.sheet_names

dfs={}

for sheet_name in sheet_names:
    df=pd.read_excel(stock_Data_path,sheet_name=sheet_name)
    dfs[sheet_name]=df
 

# %%
Consol_data=pd.DataFrame()

for i in range(len(sheet_names)):
    country_name=sheet_names[i].replace(' ','_').split('_-')[0]
    stock_name= sheet_names[i].replace(' ','_').split('-_')[1].split('_(')[0]
    
    country= dfs[sheet_names[i]].iloc[:-1,:].copy()
   
    country['Country'] = country_name
    country['Stock'] = stock_name

    Consol_data = pd.concat([Consol_data, country])

    data_order=['Stock','Country','Date', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Price']
    Consol_data=Consol_data[data_order]
    Consol_data['Date']=pd.to_datetime(Consol_data['Date'])

# Optional: resetting the index of Consol_data if needed
Consol_data.reset_index(drop=True, inplace=True)

# %% [markdown]
# # LSTM Model

# %%
# Defining the LSTM model architecture
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',run_eagerly=True)
    return model

# %%
# Defining the LSTM split sequence
def split_sequence(sequence, n_steps):
    X, Y = [], []

    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        Y.append(seq_y)

    return np.array(X), np.array(Y)

# %%
def Train_LSTM(Raw_data, Stock, n_steps):
    # Filter data for the current country
    country_data = Raw_data[Raw_data['Stock'] == Stock]
        
    # Extract the price data for the current country
    price_data = country_data['Price'].values.astype(float)
        
    # Normalize the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    price_data_scaled = scaler.fit_transform(price_data.reshape(-1, 1))
    price_data_scaled=pd.DataFrame(price_data_scaled.reshape(-1))

    country_data.reset_index(drop=True, inplace=True)
    data_concat=pd.concat([country_data,price_data_scaled],axis=1)
        
    Train_data_scaled=data_concat[data_concat['Date']<='2020-12-31']
    Test_data_scaled=data_concat[data_concat['Date']>'2020-12-31']

    Train_price_scaled=Train_data_scaled.iloc[:,-1]
    Train_price_scaled=np.array(Train_price_scaled)

        
    # Split the sequence for the current country
    X_train_country, Y_train_country = split_sequence(Train_price_scaled, n_steps)

    # Reshape the input data
    X_train_country = X_train_country.reshape((X_train_country.shape[0], n_steps, 1))

    # Train the LSTM model
    model = create_lstm_model((X_train_country.shape[1], 1))
    model.fit(X_train_country, Y_train_country, epochs=10, batch_size=2, verbose=0)
    
    return Test_data_scaled, model, scaler, n_steps, Train_data_scaled

# %%
def Predict_LSTM(Test_data, model, scaler, n_steps):

    Test_price_scaled=Test_data.iloc[:,-1]
    Test_price_scaled=np.array(Test_price_scaled)

    X_test_country, Y_test_country = split_sequence(Test_price_scaled, n_steps)
    X_test_country = X_test_country.reshape((X_test_country.shape[0], n_steps, 1))

    test_predict= model.predict(X_test_country)
    test_predict= np.round(scaler.inverse_transform(test_predict))

    Predicted_data=Test_data.iloc[n_steps:,[0,1,2,-2]]
    Predicted_data['Test_predicted']=test_predict
    
    return Predicted_data

# %%
def Calculate_bands_and_plot(Train_data, test_and_predicted_data,Stock):

    Bband_data=pd.concat([Train_data,test_and_predicted_data])
    Bband_data=Bband_data.sort_values(by='Date',ascending=True)
    Bband_data=Bband_data.iloc[:,2:]
    Bband_data['Date']=[pd.to_datetime(date) for date in Bband_data['Date']]

    Bband_data['SMA']=np.round(Bband_data['Price'].rolling(window=5).mean(),2)
    Bband_data['STD']=np.round(Bband_data['Price'].rolling(window=5).std(),2)

    Bband_data['Upper_Band']=Bband_data['SMA']+(Bband_data['STD']*2)
    Bband_data['Lower_Band']=Bband_data['SMA']-(Bband_data['STD']*2)

    Bband_data['Stock']=Stock

    plt.figure(figsize=(14,6))
    plt.title(Stock)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(Bband_data['Date'],Bband_data['Price'])
    plt.plot(Bband_data['Date'],Bband_data['Test_predicted'])
    plt.plot(Bband_data['Date'],Bband_data['Upper_Band'], color='grey',linestyle='--')
    plt.plot(Bband_data['Date'],Bband_data['Lower_Band'], color='grey',linestyle='--')

    plt.fill_between(Bband_data['Date'],Bband_data['Upper_Band'],Bband_data['Lower_Band'],color='grey',alpha=0.2)
    plt.legend(Bband_data[['Price','Test_predicted','Upper_Band','Lower_Band']], loc='lower right')
    plt.show()
    
    return Bband_data

# %%
def Make_decision(Bband_data,Stock):
    Bband_data = Bband_data[Bband_data['Stock'] == Stock]
    Bband_data = Bband_data.copy()
    Decision=[]
    for index, row in Bband_data.iterrows():
        if row['Test_predicted'] > row['Upper_Band']:
            Decision.append("Sell")
        elif row['Test_predicted'] < row['Lower_Band']:
            Decision.append("Buy")
        elif np.isnan(row['Test_predicted']):
            Decision.append("N/A")    
        else:
            Decision.append("Hold")

    Bband_data['Decision']= np.array(Decision)
    #Bband_data.loc[:,'Decision']=np.array(Decision)
    return Bband_data[['Stock','Date','Price','Test_predicted','Decision']]

# %%
def Calculate_Equity_Curve(All_Bband_data, Stock):
    Test_data_extract= All_Bband_data[All_Bband_data['Date']>'2020-12-31']
    Test_data_extract=Test_data_extract[Test_data_extract['Stock']==Stock]
    Test_data_extract['Position']=None
    Test_data_extract['Position'] = np.where(Test_data_extract['Test_predicted'] < Test_data_extract['Lower_Band'], 1, 0)
    Test_data_extract['Position'] = np.where(Test_data_extract['Test_predicted'] > Test_data_extract['Upper_Band'], -1, Test_data_extract['Position'])
    Test_data_extract['Returns'] = 1+(Test_data_extract['Test_predicted'].pct_change()*Test_data_extract['Position'].shift(1))
    Test_data_extract['Cumulative_Returns'] = Test_data_extract['Returns'].cumprod()

    #print("Test Set RMSE: ",np.round(math.sqrt(mean_squared_error(All_Country_Predicted['Price'],All_Country_Predicted['Test_predicted'])),3))
    Stock_MAPE=np.round(mean_absolute_percentage_error(Test_data_extract['Price'],Test_data_extract['Test_predicted'])*100,2)

    plt.plot(Test_data_extract['Date'],Test_data_extract['Cumulative_Returns'])
    plt.title(Stock+' '+'Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('CumulativeReturns (%)')
    # Formatting the x-axis labels to show only day and month
    date_formatter = DateFormatter('%d-%m')  
    plt.gca().xaxis.set_major_formatter(date_formatter)  
    plt.show()

    return print(Stock+' '+'MAPE:',Stock_MAPE), print('Cumulative return: ', np.round(Test_data_extract['Cumulative_Returns'].iloc[-1],2),'%')

# %%
# Calculating and plotting curves for all the stocks. 

# Calculating and plotting LSTM model performance for first 3 stocks
Stocks=Consol_data['Stock'].unique()
All_Country_Predicted= pd.DataFrame()
All_Bband_data= pd.DataFrame()

for Stock in Stocks:
    Test_data_scaled,  model, scaler, n_steps, Train_data_scaled = Train_LSTM(Consol_data, Stock,n_steps=5)
    
    Predicted_data = Predict_LSTM(Test_data_scaled,model,scaler,n_steps)

    All_Country_Predicted=pd.concat([All_Country_Predicted,Predicted_data])
    
    Bband_data= Calculate_bands_and_plot(Train_data_scaled.iloc[:,[0,1,2,-2]],Predicted_data,Stock)

    All_Bband_data=pd.concat([All_Bband_data,Bband_data])
    
    print(Make_decision(All_Bband_data,Stock).tail())
    
    Calculate_Equity_Curve(All_Bband_data, Stock)


