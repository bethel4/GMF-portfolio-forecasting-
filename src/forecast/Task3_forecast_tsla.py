# task3_forecast_tsla.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Uncomment depending on your model type
from statsmodels.tsa.statespace.sarimax import SARIMAX  # for SARIMA
# from keras.models import load_model                     # for LSTM
# from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 1: Load Tesla Historical Data
# -----------------------------
data_path = 'data/TSLA_historical_data.csv'
tesla_df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
tesla_df = tesla_df.sort_index()

# -----------------------------
# Step 2A: Forecast using SARIMA
# -----------------------------
use_sarima = True  # Set False if you want to use LSTM

if use_sarima:
    # Define SARIMA model (adjust order and seasonal_order based on Task 2 training)
    sarima_model = SARIMAX(tesla_df['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_results = sarima_model.fit(disp=False)
    
    # Forecast 12 months
    forecast_steps = 12
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Plot forecast
    plt.figure(figsize=(12,6))
    plt.plot(tesla_df['Close'], label='Historical')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='orange')
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:,0],
                     forecast_ci.iloc[:,1],
                     color='orange', alpha=0.3)
    plt.title('Tesla Stock Price Forecast (SARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tsla_forecast_sarima.png', dpi=300)
    plt.show()

# -----------------------------
# Step 2B: Forecast using LSTM
# -----------------------------
else:
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler

    lstm_model = load_model('models/tsla_lstm_model.h5')

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(tesla_df['Close'].values.reshape(-1,1))

    # Prepare last 60 days for prediction
    look_back = 60
    last_sequence = scaled_data[-look_back:]
    X_test = np.array([last_sequence])

    # Predict next 12 days (or months if monthly aggregation)
    predicted_scaled = lstm_model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)

    # Plot forecast
    plt.figure(figsize=(12,6))
    plt.plot(tesla_df['Close'], label='Historical')
    forecast_dates = pd.date_range(tesla_df.index[-1], periods=13, freq='M')[1:]
    plt.plot(forecast_dates, predicted.flatten(), label='Forecast', color='orange')
    plt.title('Tesla Stock Price Forecast (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tsla_forecast_lstm.png', dpi=300)
    plt.show()
