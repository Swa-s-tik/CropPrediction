import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to load the data
@st.cache_data
def load_data():
    file_path = r'C:\Users\swastik\Desktop\crop_pred\crop_yield.csv'
    return pd.read_csv(file_path)

# Function to plot the heatmap
def plot_heatmap(data):
    numeric_columns = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Function to plot production trends
def plot_production_trends(data, crop, state):
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Crop_Year'], filtered_data['Production'], marker='o')
    plt.title(f'{crop} Production in {state} (1997-2017)')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.grid(True)
    st.pyplot(plt)

# Function to prepare data for LSTM
def prepare_data_for_lstm(data, lookback=3):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Function to forecast production using LSTM
def forecast_production_lstm(data, crop, state):
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    
    # Prepare data for LSTM
    production_data = filtered_data['Production'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    production_scaled = scaler.fit_transform(production_data)
    
    # Prepare data with lookback
    lookback = 3
    X, y = prepare_data_for_lstm(production_scaled, lookback)
    
    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], y_train, label='Actual (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], y_test, label='Actual (Test)')
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], train_predict, label='Predicted (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], test_predict, label='Predicted (Test)')
    plt.title(f'LSTM Forecast - {crop} Production in {state}')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.legend()
    st.pyplot(plt)
    
    # Display RMSE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    
    # Make future predictions
    last_sequence = production_scaled[-lookback:]
    future_predictions = []
    
    for _ in range(5):  # Predict next 5 years
        next_pred = model.predict(last_sequence.reshape(1, lookback, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Display future predictions
    st.subheader("Future Predictions")
    future_years = pd.date_range(start=str(filtered_data['Crop_Year'].max() + 1), periods=5, freq='Y')
    future_df = pd.DataFrame({'Year': future_years.year, 'Predicted Production': future_predictions.flatten()})
    st.write(future_df)

# Function to analyze top crops and states
def analyze_top_producers(data):
    # Top 5 crops by total production
    top_crops = data.groupby('Crop')['Production'].sum().sort_values(ascending=False).head()
    st.subheader("Top 5 Crops by Total Production")
    st.bar_chart(top_crops)
    
    # Top 5 states by total production
    top_states = data.groupby('State')['Production'].sum().sort_values(ascending=False).head()
    st.subheader("Top 5 States by Total Production")
    st.bar_chart(top_states)

# Main function for the Streamlit app
def main():
    st.title("Crop Production Analysis and Forecasting")
    
    data = load_data()
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())

    st.subheader("Correlation Heatmap")
    plot_heatmap(data)

    analyze_top_producers(data)

    st.subheader("Production Trends and Forecast")
    crop = st.selectbox("Select Crop", data['Crop'].unique())
    state = st.selectbox("Select State", data['State'].unique())
    
    plot_production_trends(data, crop, state)
    forecast_production_lstm(data, crop, state)

if __name__ == "__main__":
    main()
