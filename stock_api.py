# stock_api.py

import os
from fastapi import FastAPI, Query
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



app = FastAPI()
alpha_vantage_api_key = "A8HV31BOR18LV66O"
ts = TimeSeries(key=alpha_vantage_api_key, output_format="pandas")

async def get_stock_data(symbol: str):
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize="full")
    return data

def arima_prediction(data):
    model = auto_arima(data, suppress_warnings=True)
    forecast = model.predict(n_periods=5)
    return forecast
def lstm_prediction(data):
    # Prepare the data
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Train-test split
    train_data, test_data = train_test_split(data_scaled, train_size=0.8, shuffle=False)
    
    # Create TimeseriesGenerator for training data
    look_back = 10
    train_generator = TimeseriesGenerator(train_data, train_data, length=look_back, batch_size=1)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation="relu", input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(train_generator, epochs=50, verbose=0)

    # Make predictions
    # Create TimeseriesGenerator for test data
    test_generator = TimeseriesGenerator(test_data, test_data, length=look_back, batch_size=1)
    
    predictions = model.predict(test_generator)
    
    # Inverse scaling
    predictions = scaler.inverse_transform(predictions)
    
    return predictions[-5:]



def prophet_prediction(data):
    df = data.reset_index().rename(columns={"index": "ds", "4. close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast["yhat"][-5:].values

@app.get("/stock/{symbol}")
async def get_stock(symbol: str, method: str = Query("arima", enum=["arima", "lstm", "prophet"])):
    data = await get_stock_data(symbol)
    if method == "arima":
        predictions = arima_prediction(data["4. close"])
    elif method == "lstm":
        predictions = lstm_prediction(data["4. close"])
    else:
        predictions = prophet_prediction(data["4. close"])
    return {"symbol": symbol, "predictions": predictions.tolist()}
