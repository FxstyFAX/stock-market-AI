import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_loader import load_data
from utils.indicators import add_technical_indicators
from utils.model import create_lstm_model
from utils.auth import require_login
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# User login
require_login()

st.title("📈 Stock Price Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Load and display data
data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data = add_technical_indicators(data)
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(data.tail())

# Prepare data for LSTM
features = ['Close', 'RSI', 'MACD', 'SMA', 'EMA', 'BB_High', 'BB_Low']
data_filtered = data[features]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_filtered)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the model
model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]
actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1,1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]

# Plot predictions
st.subheader('Predicted vs Actual Prices')
fig, ax = plt.subplots()
ax.plot(actual, label='Actual Price')
ax.plot(predictions, label='Predicted Price')
ax.legend()
st.pyplot(fig)

# Download predictions
pred_df = pd.DataFrame({'Actual': actual, 'Predicted': predictions})
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name='predictions.csv',
    mime='text/csv',
)

# Logout
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()
