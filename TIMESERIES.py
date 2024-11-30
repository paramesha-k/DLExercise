import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your time series data (assuming it's a single column of values)
# Replace this with your actual data source
data = pd.read_csv('Electric_Production.csv')
time_series = data['value'].values.reshape(-1, 1)

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
time_series_normalized = scaler.fit_transform(time_series)
print(time_series_normalized)

# Split the data into training and test sets
train_size = int(len(time_series_normalized) * 0.8)
train_data = time_series_normalized[:train_size]
test_data = time_series_normalized[train_size:]

# Create sequences for training data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10  # Adjust as needed
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the predictions and the actual values to their original scale
y_pred_original_scale = scaler.inverse_transform(y_pred)
y_test_original_scale = scaler.inverse_transform(y_test)

# Calculate RMSE (Root Mean Squared Error) as a performance metric
rmse = sqrt(mean_squared_error(y_test_original_scale, y_pred_original_scale))
print(f"Root Mean Squared Error: {rmse}")

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_original_scale, label='Actual')
plt.plot(y_pred_original_scale, label='Predicted')
plt.legend()
plt.title('Time Series Forecasting with LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
