import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

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
def prepare_data_for_lstm(data, lookback=5):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback, 0])
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
    lookback = 5
    X, y = [], []
    for i in range(len(production_scaled) - lookback):
        X.append(production_scaled[i:(i + lookback), 0])
        y.append(production_scaled[i + lookback, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE and MAPE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    train_mape = np.mean(np.abs((y_train - train_predict) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - test_predict) / y_test)) * 100
    
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
    
    # Display RMSE and MAPE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    st.write(f'Train MAPE: {train_mape:.2f}%')
    st.write(f'Test MAPE: {test_mape:.2f}%')
    
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
    future_years = pd.date_range(start=str(filtered_data['Crop_Year'].max() + 1), periods=5, freq='YE')
    future_df = pd.DataFrame({'Year': future_years.year, 'Predicted Production': future_predictions.flatten()})
    st.write(future_df)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)    
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    
    # Feature engineering
    filtered_data['Yield'] = filtered_data['Production'] / filtered_data['Area']
    
    # Prepare data for LSTM
    features = ['Production', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    lstm_data = filtered_data[features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    lstm_data_scaled = scaler.fit_transform(lstm_data)
    
    # Prepare data with lookback
    lookback = 5
    X, y = prepare_data_for_lstm(lstm_data_scaled, lookback)
    
    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(lookback, len(features)), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(50, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict_original = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_train_original = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], len(features)-1))), axis=1))[:, 0]
    test_predict_original = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_test_original = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))), axis=1))[:, 0]
    
    # Calculate RMSE and MAPE
    train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict_original))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict_original))
    train_mape = mean_absolute_percentage_error(y_train_original, train_predict_original)
    test_mape = mean_absolute_percentage_error(y_test_original, test_predict_original)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], y_train_original, label='Actual (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], y_test_original, label='Actual (Test)')
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], train_predict_original, label='Predicted (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], test_predict_original, label='Predicted (Test)')
    plt.title(f'LSTM Forecast - {crop} Production in {state}')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.legend()
    st.pyplot(plt)
    
    # Display RMSE and MAPE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    st.write(f'Train MAPE: {train_mape:.2%}')
    st.write(f'Test MAPE: {test_mape:.2%}')
    
    # Make future predictions
    last_sequence = lstm_data_scaled[-lookback:]
    future_predictions = []
    
    for _ in range(5):  # Predict next 5 years
        next_pred = model.predict(last_sequence.reshape(1, lookback, len(features)))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred[0]
    
    future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((5, len(features)-1))), axis=1))[:, 0]
    
    # Display future predictions
    st.subheader("Future Predictions")
    future_years = pd.date_range(start=str(filtered_data['Crop_Year'].max() + 1), periods=5, freq='YE')
    future_df = pd.DataFrame({'Year': future_years.year, 'Predicted Production': future_predictions})
    st.write(future_df)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)    
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    
    # Feature engineering
    filtered_data['Yield'] = filtered_data['Production'] / filtered_data['Area']
    
    # Prepare data for LSTM
    features = ['Production', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    lstm_data = filtered_data[features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    lstm_data_scaled = scaler.fit_transform(lstm_data)
    
    # Prepare data with lookback
    lookback = 5
    X, y = prepare_data_for_lstm(lstm_data_scaled, lookback)
    
    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(lookback, len(features)), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(50, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict_original = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_train_original = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], len(features)-1))), axis=1))[:, 0]
    test_predict_original = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_test_original = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))), axis=1))[:, 0]
    
    # Calculate RMSE and MAPE
    train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict_original))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict_original))
    train_mape = mean_absolute_percentage_error(y_train_original, train_predict_original)
    test_mape = mean_absolute_percentage_error(y_test_original, test_predict_original)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], y_train_original, label='Actual (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], y_test_original, label='Actual (Test)')
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], train_predict_original, label='Predicted (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], test_predict_original, label='Predicted (Test)')
    plt.title(f'LSTM Forecast - {crop} Production in {state}')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.legend()
    st.pyplot(plt)
    
    # Display RMSE and MAPE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    st.write(f'Train MAPE: {train_mape:.2%}')
    st.write(f'Test MAPE: {test_mape:.2%}')
    
    # Make future predictions
    last_sequence = lstm_data_scaled[-lookback:]
    future_predictions = []
    
    for _ in range(5):  # Predict next 5 years
        next_pred = model.predict(last_sequence.reshape(1, lookback, len(features)))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = np.concatenate((next_pred, np.zeros((1, len(features)-1))), axis=1)[0]
    
    future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((5, len(features)-1))), axis=1))[:, 0]
    
    # Display future predictions
    st.subheader("Future Predictions")
    future_years = pd.date_range(start=str(filtered_data['Crop_Year'].max() + 1), periods=5, freq='Y')
    future_df = pd.DataFrame({'Year': future_years.year, 'Predicted Production': future_predictions})
    st.write(future_df)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    
    # Feature engineering
    filtered_data['Yield'] = filtered_data['Production'] / filtered_data['Area']
    
    # Prepare data for LSTM
    features = ['Production', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    lstm_data = filtered_data[features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    lstm_data_scaled = scaler.fit_transform(lstm_data)
    
    # Prepare data with lookback
    lookback = 5
    X, y = prepare_data_for_lstm(lstm_data_scaled, lookback)
    
    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(lookback, len(features)), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(50, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict_original = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_train_original = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], len(features)-1))), axis=1))[:, 0]
    test_predict_original = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:, 0]
    y_test_original = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))), axis=1))[:, 0]
    
    # Calculate RMSE and MAPE
    train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict_original))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict_original))
    train_mape = mean_absolute_percentage_error(y_train_original, train_predict_original)
    test_mape = mean_absolute_percentage_error(y_test_original, test_predict_original)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], y_train_original, label='Actual (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], y_test_original, label='Actual (Test)')
    plt.plot(filtered_data['Crop_Year'][lookback:train_size+lookback], train_predict_original, label='Predicted (Train)')
    plt.plot(filtered_data['Crop_Year'][train_size+lookback:], test_predict_original, label='Predicted (Test)')
    plt.title(f'LSTM Forecast - {crop} Production in {state}')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.legend()
    st.pyplot(plt)
    
    # Display RMSE and MAPE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    st.write(f'Train MAPE: {train_mape:.2%}')
    st.write(f'Test MAPE: {test_mape:.2%}')
    
    # Make future predictions
    last_sequence = lstm_data_scaled[-lookback:]
    future_predictions = []
    
    for _ in range(5):  # Predict next 5 years
        next_pred = model.predict(last_sequence.reshape(1, lookback, len(features)))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = np.concatenate((next_pred, np.zeros((1, len(features)-1))), axis=1)
    
    future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((5, len(features)-1))), axis=1))[:, 0]
    
    # Display future predictions
    st.subheader("Future Predictions")
    future_years = pd.date_range(start=str(filtered_data['Crop_Year'].max() + 1), periods=5, freq='Y')
    future_df = pd.DataFrame({'Year': future_years.year, 'Predicted Production': future_predictions})
    st.write(future_df)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

# Display RMSE
    st.write(f'Train RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')
    
    # Make future predictions
    last_sequence = lstm_data_scaled[-lookback:]
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