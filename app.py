import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function to load the data
@st.cache_data
def load_data():
    file_path = 'crop_yield.csv'  # Update this path as needed
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

# Function to forecast production using Random Forest
def forecast_production_rf(data, crop, state):
    filtered_data = data[(data['Crop'] == crop) & (data['State'] == state)]
    filtered_data = filtered_data.sort_values(by='Crop_Year')
    
    # Prepare data for Random Forest
    X = filtered_data[['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = filtered_data['Production']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Calculate RMSE and MAPE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    train_mape = mean_absolute_percentage_error(y_train, train_predict)
    test_mape = mean_absolute_percentage_error(y_test, test_predict)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train['Crop_Year'], y_train, label='Actual (Train)', alpha=0.7)
    plt.scatter(X_test['Crop_Year'], y_test, label='Actual (Test)', alpha=0.7)
    plt.scatter(X_train['Crop_Year'], train_predict, label='Predicted (Train)', alpha=0.7)
    plt.scatter(X_test['Crop_Year'], test_predict, label='Predicted (Test)', alpha=0.7)
    plt.title(f'Random Forest Forecast - {crop} Production in {state}')
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
    # Make future predictions
    future_years = pd.DataFrame({'Crop_Year': range(filtered_data['Crop_Year'].max() + 1, filtered_data['Crop_Year'].max() + 6)})
    last_row = filtered_data[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']].tail(1)
    future_data = pd.concat([future_years, pd.concat([last_row] * 5, ignore_index=True)], axis=1)
    future_predictions = model.predict(future_data)
    
    # Display future predictions
    st.subheader("Future Predictions")
    future_df = pd.DataFrame({'Year': future_data['Crop_Year'], 'Predicted Production': future_predictions})
    st.write(future_df)

    # Plot feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importances')
    plt.tight_layout()
    st.pyplot(plt)

# Main function for the Streamlit app
def main():
    st.title("CropCast: Harvesting Tomorrow's Yields Today")
    
    data = load_data()
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())

    st.subheader("Correlation Heatmap")
    plot_heatmap(data)

    st.subheader("Production Trends and Forecast")
    crop = st.selectbox("Select Crop", data['Crop'].unique())
    state = st.selectbox("Select State", data['State'].unique())
    
    plot_production_trends(data, crop, state)
    forecast_production_rf(data, crop, state)

if __name__ == "__main__":
    main()
