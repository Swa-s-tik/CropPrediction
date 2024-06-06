import streamlit as st
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools

# Function to load the data
@st.cache_data
def load_data():
    file_path = "https://raw.githubusercontent.com/Swa-s-tik/CropPrediction/main/crop_yield.csv"
    return pd.read_csv(file_path)

# Function to plot the heatmap
#def plot_heatmap(data):
 #   numeric_columns = data.select_dtypes(include=[np.number])
  #  correlation_matrix = numeric_columns.corr()
   # plt.figure(figsize=(12, 8))
    #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    #st.pyplot(plt.gcf())

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
    st.pyplot(plt.gcf())

# Function to find the best ARIMA parameters
def find_best_arima_params(train):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_param = None
    for param in pdq:
        try:
            temp_model = ARIMA(train, order=param)
            result = temp_model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_param = param
        except:
            continue
    return best_param

# Function to forecast production using ARIMA
def forecast_production(data, crop, state):
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    filtered_data.set_index('Crop_Year', inplace=True)
    production_data = filtered_data['Production']

    # Log transformation to stabilize variance
    production_data_log = np.log(production_data)
    
    # Train-test split
    train_size = int(len(production_data_log) * 0.8)
    train, test = production_data_log[:train_size], production_data_log[train_size:]

    # Find the best ARIMA parameters
    best_param = find_best_arima_params(train)

    # Fit the ARIMA model on the training set
    model = ARIMA(train, order=best_param)
    model_fit = model.fit()

    # Forecast the production values on the test set
    forecast_log = model_fit.forecast(steps=len(test))
    forecast = np.exp(forecast_log)  # Reverse the log transformation

    # Evaluate the model's performance
    mse = mean_squared_error(np.exp(test), forecast)
    rmse = np.sqrt(mse)

    # Plot the actual vs forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, np.exp(train), label='Train')
    plt.plot(test.index, np.exp(test), label='Test')
    plt.plot(test.index, forecast, label='Forecast', linestyle='--')
    plt.title(f'ARIMA Model - {crop} Production Forecast in {state}')
    plt.xlabel('Year')
    plt.ylabel('Production (in metric tons)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

    st.write(f'RMSE: {rmse}')

# Main function for the Streamlit app
def main():
    st.title("Crop Production Analysis and Forecasting")
    
    data = load_data()
    st.subheader("Dataset")
    st.write(data.head())

    #st.subheader("Correlation Heatmap")
    #plot_heatmap(data)

    st.subheader("Production Trends")
    crop = st.selectbox("Select Crop", data['Crop'].unique())
    state = st.selectbox("Select State", data['State'].unique())
    plot_production_trends(data, crop, state)

    st.subheader("Production Forecast")
    forecast_production(data, crop, state)

if __name__ == "__main__":
    main()
