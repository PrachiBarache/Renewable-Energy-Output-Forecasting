"""
Renewable Energy Output Forecasting Project
===========================================
This project develops meteo-to-power forecasting models to predict solar and wind energy
generation based on weather data using various time series forecasting techniques.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import io
import zipfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Prophet is optional - we'll have fallback methods
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create directories for saving data and models
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

#############################################################
# PART 1: DATA ACQUISITION
#############################################################

import requests

def download_weather_data():
    """
    Download weather data from NREL's NSRDB with better error handling.
    """
    print("Attempting to download weather data...")
    
    # For this example, we'll use a sample dataset from NREL's website
    url = "https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT(-104.5733642578125%2039.75391559870632)&names=2019&leap_day=false&interval=60&utc=false&email=info%40example.com&api_key=DEMO_KEY"
    
    try:
        print(f"Sending request to: {url}")
        response = requests.get(url)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        if response.status_code != 200:
            print(f"Error response content: {response.text[:500]}...")  # Show first 500 chars
        
        if response.status_code == 200:
            # Skip the first 2 rows which contain metadata
            weather_data = pd.read_csv(io.StringIO(response.text), skiprows=2)
            weather_data.to_csv('data/raw/weather_data_sample.csv', index=False)
            print(f"Weather data saved to 'data/raw/weather_data_sample.csv'")
            return weather_data
        else:
            print(f"Failed to download weather data. Status code: {response.status_code}")
            # Fallback to creating synthetic data
            return create_synthetic_weather_data()
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        # Fallback to creating synthetic data
        return create_synthetic_weather_data()
    
import requests
import io
import pandas as pd

def get_opsd_data():
    """Download Open Power System Data."""
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    
    response = requests.get(url)
    if response.status_code == 200:
        # Load directly into pandas
        data = pd.read_csv(io.StringIO(response.text))
        # Save to file
        data.to_csv('power_generation_data.csv', index=False)
        print("Power generation data downloaded successfully")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
power_data = get_opsd_data()

def create_synthetic_weather_data():
    """
    Create synthetic weather data if real data cannot be downloaded.
    """
    print("Creating synthetic weather data...")
    
    # Create a daterange for one year with hourly data
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a dataframe with the date range
    weather_data = pd.DataFrame({'timestamp': date_range})
    
    # Add month and hour columns for seasonality
    weather_data['month'] = weather_data['timestamp'].dt.month
    weather_data['hour'] = weather_data['timestamp'].dt.hour
    weather_data['day_of_year'] = weather_data['timestamp'].dt.dayofyear
    
    # Generate synthetic weather features
    
    # Temperature: seasonal pattern with daily fluctuations
    seasonal_temp = 15 * np.sin(2 * np.pi * (weather_data['day_of_year'] - 80) / 365) + 10  # Yearly cycle
    daily_temp = 5 * np.sin(2 * np.pi * weather_data['hour'] / 24)  # Daily cycle
    weather_data['temperature'] = seasonal_temp + daily_temp + np.random.normal(0, 2, len(weather_data))
    
    # Wind speed: more random but with some seasonality
    weather_data['wind_speed'] = 5 + 3 * np.sin(2 * np.pi * weather_data['day_of_year'] / 365) + np.random.gamma(2, 2, len(weather_data))
    
    # Solar radiation: strong daily pattern, zero at night, seasonal variation
    day_factor = np.sin(np.pi * (weather_data['hour'] - 6) / 12) * (weather_data['hour'] >= 6) * (weather_data['hour'] <= 18)
    season_factor = 1 + 0.5 * np.sin(2 * np.pi * (weather_data['day_of_year'] - 172) / 365)  # Peak in summer
    weather_data['solar_radiation'] = 800 * day_factor * season_factor + np.random.normal(0, 50, len(weather_data))
    weather_data['solar_radiation'] = weather_data['solar_radiation'].clip(0)
    
    # Cloud cover: inverse correlation with solar radiation during daytime
    cloud_base = 0.5 - 0.3 * np.sin(2 * np.pi * (weather_data['day_of_year'] - 172) / 365)  # More clouds in winter
    cloud_daily = day_factor * -0.3 + 0.3  # Less clouds at mid-day
    weather_data['cloud_cover'] = (cloud_base + cloud_daily + np.random.normal(0, 0.2, len(weather_data))).clip(0, 1)
    
    # Save to CSV
    weather_data.to_csv('data/raw/weather_data_synthetic.csv', index=False)
    print(f"Synthetic weather data saved to 'data/raw/weather_data_synthetic.csv'")
    
    return weather_data

def create_synthetic_power_data():
    """
    Create synthetic power generation data if real data cannot be downloaded.
    """
    print("Creating synthetic power generation data...")
    
    # Create a daterange for one year with hourly data
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a dataframe with the date range
    power_data = pd.DataFrame({'timestamp': date_range})
    
    # Add month and hour columns for seasonality
    power_data['month'] = power_data['timestamp'].dt.month
    power_data['hour'] = power_data['timestamp'].dt.hour
    power_data['day_of_year'] = power_data['timestamp'].dt.dayofyear
    
    # Generate synthetic power generation
    
    # Solar power: depends on hour of day and season
    day_factor = np.sin(np.pi * (power_data['hour'] - 6) / 12) * (power_data['hour'] >= 6) * (power_data['hour'] <= 18)
    season_factor = 1 + 0.7 * np.sin(2 * np.pi * (power_data['day_of_year'] - 172) / 365)  # Peak in summer
    cloud_noise = np.random.beta(2, 5, len(power_data)) * 0.5  # Simulates cloud cover
    power_data['solar_generation'] = 100 * day_factor * season_factor * (1 - cloud_noise)
    
    # Wind power: more random but with some patterns
    wind_base = 40 + 20 * np.sin(2 * np.pi * power_data['day_of_year'] / 365)  # Seasonal pattern
    wind_noise = np.random.gamma(2, 4, len(power_data))
    wind_autocorr = np.zeros(len(power_data))
    
    # Add autocorrelation to wind (wind tends to stay similar for periods)
    rho = 0.8
    sigma = 10
    for i in range(1, len(power_data)):
        wind_autocorr[i] = rho * wind_autocorr[i-1] + np.random.normal(0, sigma)
    
    power_data['wind_generation'] = wind_base + wind_autocorr + wind_noise
    power_data['wind_generation'] = power_data['wind_generation'].clip(0)
    
    # Save to CSV
    power_data.to_csv('data/raw/power_generation_synthetic.csv', index=False)
    print(f"Synthetic power generation data saved to 'data/raw/power_generation_synthetic.csv'")
    
    return power_data

def download_power_generation_data():
    """Download power generation data from Open Power System Data."""
    print("Downloading power generation data...")
    
    # Latest package URL (check website for updates)
    latest_package_url = "https://data.open-power-system-data.org/time_series/2023-06-20/time_series_60min_singleindex.csv"
    
    # try:
    #     response = requests.get(latest_package_url)
    #     # Rest of your function remains the same

#############################################################
# PART 2: DATA PREPROCESSING
#############################################################

def preprocess_weather_data(weather_data):
    """
    Preprocess the weather data.
    """
    print("Preprocessing weather data...")
    
    # Check if we're using the NREL data or synthetic data
    if 'Year' in weather_data.columns:
        # NREL data format
        # Rename columns to more readable names
        weather_data = weather_data.rename(columns={
            'Year': 'year',
            'Month': 'month',
            'Day': 'day',
            'Hour': 'hour',
            'Minute': 'minute',
            'Temperature': 'temperature',
            'Wind Speed': 'wind_speed',
            'GHI': 'solar_radiation',
            'DHI': 'diffuse_radiation',
            'DNI': 'direct_radiation',
            'Cloud Type': 'cloud_type',
            'Dew Point': 'dew_point',
            'Relative Humidity': 'humidity'
        })
        
        # Create timestamp column
        weather_data['timestamp'] = pd.to_datetime(
            weather_data[['year', 'month', 'day', 'hour', 'minute']]
        )
        
    elif 'timestamp' in weather_data.columns:
        # Ensure timestamp is in datetime format
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    
    # Set timestamp as index
    weather_data = weather_data.set_index('timestamp')
    
    # Select relevant features
    relevant_features = [col for col in weather_data.columns if col in [
        'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover',
        'diffuse_radiation', 'direct_radiation', 'dew_point', 'humidity'
    ]]
    
    weather_processed = weather_data[relevant_features].copy()
    
    # Handle missing values
    weather_processed = weather_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Add engineered features
    
    # Time-based features
    weather_processed['hour'] = weather_processed.index.hour
    weather_processed['day_of_year'] = weather_processed.index.dayofyear
    weather_processed['month'] = weather_processed.index.month
    weather_processed['day_of_week'] = weather_processed.index.dayofweek
    
    # Cyclical encoding of time features
    weather_processed['hour_sin'] = np.sin(2 * np.pi * weather_processed['hour'] / 24)
    weather_processed['hour_cos'] = np.cos(2 * np.pi * weather_processed['hour'] / 24)
    weather_processed['day_sin'] = np.sin(2 * np.pi * weather_processed['day_of_year'] / 365)
    weather_processed['day_cos'] = np.cos(2 * np.pi * weather_processed['day_of_year'] / 365)
    weather_processed['month_sin'] = np.sin(2 * np.pi * weather_processed['month'] / 12)
    weather_processed['month_cos'] = np.cos(2 * np.pi * weather_processed['month'] / 12)
    
    # Save processed data
    weather_processed.to_csv('data/processed/weather_processed.csv')
    print(f"Processed weather data saved to 'data/processed/weather_processed.csv'")
    
    return weather_processed

def preprocess_power_data(power_data):
    """
    Preprocess the power generation data.
    """
    print("Preprocessing power generation data...")
    
    # Check if we're using the Open Power System data or synthetic data
    if 'utc_timestamp' in power_data.columns:
        # Open Power System data format
        power_data = power_data.rename(columns={'utc_timestamp': 'timestamp'})
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in power_data.columns:
        power_data['timestamp'] = pd.to_datetime(power_data['timestamp'])
        power_data = power_data.set_index('timestamp')
    
    # Identify solar and wind columns
    solar_cols = [col for col in power_data.columns if 'solar' in col.lower()]
    wind_cols = [col for col in power_data.columns if 'wind' in col.lower()]
    
    # If no specific columns found, use the synthetic data columns
    if not solar_cols:
        solar_cols = [col for col in power_data.columns if col in ['solar_generation']]
    if not wind_cols:
        wind_cols = [col for col in power_data.columns if col in ['wind_generation']]
    
    # Create aggregate solar and wind generation if multiple columns exist
    if solar_cols:
        power_data['solar_generation'] = power_data[solar_cols].sum(axis=1)
    
    if wind_cols:
        power_data['wind_generation'] = power_data[wind_cols].sum(axis=1)
    
    # Select only the aggregate columns
    power_processed = power_data[['solar_generation', 'wind_generation']].copy()
    
    # Handle missing values
    power_processed = power_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Add time-based features
    power_processed['hour'] = power_processed.index.hour
    power_processed['day_of_year'] = power_processed.index.dayofyear
    power_processed['month'] = power_processed.index.month
    power_processed['day_of_week'] = power_processed.index.dayofweek
    
    # Save processed data
    power_processed.to_csv('data/processed/power_processed.csv')
    print(f"Processed power data saved to 'data/processed/power_processed.csv'")
    
    return power_processed

def merge_weather_and_power_data(weather_processed, power_processed):
    """
    Merge weather and power data on timestamp index.
    """
    print("Merging weather and power data...")
    
    # Ensure both dataframes have the same datetime index format
    merged_data = pd.merge(
        weather_processed,
        power_processed,
        left_index=True,
        right_index=True,
        how='inner',
        suffixes=('_weather', '_power')
    )
    
    # Drop duplicate columns that might have been created in the merge
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    
    # Save merged data
    merged_data.to_csv('data/processed/merged_data.csv')
    print(f"Merged data saved to 'data/processed/merged_data.csv'")
    
    return merged_data

def create_lagged_features(merged_data, lag_hours=[1, 3, 6, 12, 24]):
    """
    Create lagged features for time series forecasting.
    """
    print("Creating lagged features...")
    
    df = merged_data.copy()
    
    # Add lagged features for weather variables
    for lag in lag_hours:
        for feature in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']:
            if feature in df.columns:
                df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
    
    # Add lagged features for power generation
    for lag in lag_hours:
        if 'solar_generation' in df.columns:
            df[f'solar_generation_lag_{lag}h'] = df['solar_generation'].shift(lag)
        if 'wind_generation' in df.columns:
            df[f'wind_generation_lag_{lag}h'] = df['wind_generation'].shift(lag)
    
    # Add rolling mean features
    window_sizes = [3, 6, 12, 24]
    for window in window_sizes:
        for feature in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']:
            if feature in df.columns:
                df[f'{feature}_rolling_{window}h'] = df[feature].rolling(window=window).mean()
        
        if 'solar_generation' in df.columns:
            df[f'solar_generation_rolling_{window}h'] = df['solar_generation'].rolling(window=window).mean()
        if 'wind_generation' in df.columns:
            df[f'wind_generation_rolling_{window}h'] = df['wind_generation'].rolling(window=window).mean()
    
    # Drop rows with NaN values created by lagging
    df = df.dropna()
    
    # Save feature engineered data
    df.to_csv('data/processed/feature_engineered_data.csv')
    print(f"Feature engineered data saved to 'data/processed/feature_engineered_data.csv'")
    
    return df

def prepare_train_test_data(df, target_col, forecast_horizon=24, test_size=0.2):
    """
    Prepare data for training and testing.
    
    Args:
        df: Feature engineered data
        target_col: Target column name ('solar_generation' or 'wind_generation')
        forecast_horizon: Forecast horizon in hours
        test_size: Proportion of data to use for testing
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler_X, scaler_y
    """
    print(f"Preparing train/test data for {target_col}...")
    
    # Filter out any constant or nearly constant features
    df_filtered = df.copy()
    for col in df_filtered.columns:
        if df_filtered[col].std() < 1e-6:
            df_filtered = df_filtered.drop(columns=[col])
    
    # Select features (exclude the target variables)
    feature_cols = [col for col in df_filtered.columns if col not in ['solar_generation', 'wind_generation']]
    
    # Split data chronologically (time series split)
    split_idx = int(len(df_filtered) * (1 - test_size))
    train_data = df_filtered.iloc[:split_idx].copy()
    test_data = df_filtered.iloc[split_idx:].copy()
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on training data
    X_train = scaler_X.fit_transform(train_data[feature_cols])
    y_train = scaler_y.fit_transform(train_data[[target_col]])
    
    # Transform test data
    X_test = scaler_X.transform(test_data[feature_cols])
    y_test = scaler_y.transform(test_data[[target_col]])
    
    # Save scalers for later use in predictions
    import joblib
    joblib.dump(scaler_X, f'models/scaler_X_{target_col}.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{target_col}.pkl')
    
    # Save feature names
    with open(f'models/feature_names_{target_col}.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Scalers and feature names saved in 'models/' directory")
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler_X, scaler_y

#############################################################
# PART 3: MODEL BUILDING
#############################################################

def build_classical_models(X_train, y_train, target_col):
    """
    Build and train classical ML models.
    
    Returns:
        Dictionary of trained models
    """
    print(f"Building classical models for {target_col}...")
    
    models = {}
    
    # Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear_regression'] = lr
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())
    models['random_forest'] = rf
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train.ravel())
    models['gradient_boosting'] = gb
    
    # Save models
    import joblib
    for name, model in models.items():
        joblib.dump(model, f'models/{name}_{target_col}.pkl')
    
    print(f"Classical models saved in 'models/' directory")
    
    return models

def build_arima_model(data, target_col, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
    """
    Build and train ARIMA model.
    """
    print(f"Building ARIMA model for {target_col}...")
    
    # Extract target series
    series = data[target_col].copy()
    
    # Fit SARIMAX (Seasonal ARIMA with exogenous variables)
    # For simplicity, we'll just use time of day as exogenous
    exog = pd.get_dummies(data.index.hour, prefix='hour')
    
    try:
        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit model
        results = model.fit(disp=False)
        
        # Save model
        results.save(f'models/sarimax_{target_col}.pkl')
        print(f"ARIMA model saved as 'models/sarimax_{target_col}.pkl'")
        
        return results
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        print("Trying simpler ARIMA model...")
        
        # Try a simpler model
        try:
            model = ARIMA(series, order=(1, 1, 0))
            results = model.fit()
            
            # Save model
            results.save(f'models/arima_{target_col}.pkl')
            print(f"Simpler ARIMA model saved as 'models/arima_{target_col}.pkl'")
            
            return results
        except Exception as e:
            print(f"Error fitting simpler ARIMA model: {e}")
            return None

def build_prophet_model(data, target_col):
    """
    Build and train Facebook Prophet model if available, 
    otherwise use exponential smoothing as an alternative.
    """
    if PROPHET_AVAILABLE:
        print(f"Building Prophet model for {target_col}...")
        
        # Prophet requires a specific dataframe format with 'ds' and 'y' columns
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col]
        })
        
        # Add additional regressors
        weather_features = [col for col in data.columns if col in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']]
        
        try:
            # Initialize model with daily and weekly seasonalities
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Add hourly seasonality
            model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
            
            # Add weather features as regressors
            for feature in weather_features:
                prophet_data[feature] = data[feature]
                model.add_regressor(feature)
            
            # Fit model
            model.fit(prophet_data)
            
            # Save model
            import pickle
            with open(f'models/prophet_{target_col}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Prophet model saved as 'models/prophet_{target_col}.pkl'")
            
            return model
        except Exception as e:
            print(f"Error fitting Prophet model: {e}")
            return None
    else:
        print(f"Prophet not available. Using exponential smoothing for {target_col} instead...")
        
        # Use statsmodels ExponentialSmoothing as an alternative
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Extract target series
        series = data[target_col].copy()
        
        try:
            # Initialize model with additive trend and seasonal components
            # Use 24 for hourly data seasonal period
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=24
            )
            
            # Fit model
            results = model.fit()
            
            # Save model
            import pickle
            with open(f'models/exp_smoothing_{target_col}.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Exponential smoothing model saved as 'models/exp_smoothing_{target_col}.pkl'")
            
            return results
        except Exception as e:
            print(f"Error fitting exponential smoothing model: {e}")
            
            # Fall back to simple exponential smoothing without seasonality
            try:
                model = ExponentialSmoothing(series, trend=None, seasonal=None)
                results = model.fit()
                
                import pickle
                with open(f'models/simple_exp_smoothing_{target_col}.pkl', 'wb') as f:
                    pickle.dump(results, f)
                
                print(f"Simple exponential smoothing model saved as 'models/simple_exp_smoothing_{target_col}.pkl'")
                
                return results
            except Exception as e:
                print(f"Error fitting simple exponential smoothing model: {e}")
                return None

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

def build_lstm_model(X_train, y_train, feature_names, target_col, batch_size=32, num_epochs=50):
    """
    Build and train LSTM model.
    """
    print(f"Building LSTM model for {target_col}...")
    
    # Reshape input for LSTM [batch, sequence_length, features]
    # For simplicity, we'll treat each sample as a sequence of length 1
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(X_train_reshaped, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]  # Number of features
    model = LSTMModel(input_size=input_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), f'models/lstm_{target_col}.pt')
    
    # Save feature names and architecture info
    import json
    with open(f'models/lstm_info_{target_col}.json', 'w') as f:
        json.dump({
            'input_size': input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'features': feature_names
        }, f)
    
    print(f"LSTM model saved as 'models/lstm_{target_col}.pt'")
    
    return model, losses

#############################################################
# PART 4: MODEL EVALUATION
#############################################################

def evaluate_model(model, X_test, y_test, model_name, target_col, scaler_y=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        target_col: Target column name
        scaler_y: Scaler used to transform the target
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating {model_name} for {target_col}...")
    
    if model_name in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Classical ML models
        y_pred = model.predict(X_test)
    
    elif model_name == 'lstm':
        # LSTM model
        model.eval()
        with torch.no_grad():
            # Reshape input for LSTM [batch, sequence_length, features]
            X_test_reshaped = torch.tensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), dtype=torch.float32)
            y_pred = model(X_test_reshaped).numpy()
    
    # If we have a scaler, inverse transform the predictions and actual values
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model': model_name,
        'target': target_col,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    # Save metrics
    import json
    with open(f'results/{model_name}_{target_col}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def evaluate_time_series_models(data, target_col, test_size=0.2):
    """
    Evaluate time series models (ARIMA, Prophet).
    
    Args:
        data: DataFrame with time series data
        target_col: Target column name
        test_size: Proportion of data to use for testing
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating time series models for {target_col}...")
    
    # Split data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    results = {}
    
    # Evaluate ARIMA
    try:
        print("Evaluating ARIMA model...")
        
        # Try to load SARIMAX model
        try:
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for test period
            exog_test = pd.get_dummies(test_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1],
                exog=exog_test
            ).predicted_mean
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1]
            ).predicted_mean
        
        # Calculate metrics
        y_true = test_data[target_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'model': 'arima',
            'target': target_col,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save metrics
        import json
        with open(f'results/arima_{target_col}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        results['arima'] = metrics
        
    except Exception as e:
        print(f"Error evaluating ARIMA model: {e}")
    
    # Evaluate Prophet
    try:
        print("Evaluating Prophet model...")
        
        # Load Prophet model
        import pickle
        with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare future dataframe
        future = pd.DataFrame({'ds': test_data.index})
        
        # Add weather features as regressors if they were used in training
        weather_features = [col for col in train_data.columns if col in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']]
        for feature in weather_features:
            future[feature] = test_data[feature]
        
        # Generate predictions
        forecast = model.predict(future)
        y_pred = forecast['yhat'].values
        
        # Calculate metrics
        y_true = test_data[target_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'model': 'prophet',
            'target': target_col,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save metrics
        import json
        with open(f'results/prophet_{target_col}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        results['prophet'] = metrics
        
    except Exception as e:
        print(f"Error evaluating Prophet model: {e}")
    
    return results

def plot_predictions(model, X_test, y_test, test_dates, model_name, target_col, scaler_y=None, savefig=True):
    """
    Plot model predictions against actual values.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        test_dates: Test dates (for x-axis)
        model_name: Name of the model
        target_col: Target column name
        scaler_y: Scaler used to transform the target
        savefig: Whether to save the figure to file
    """
    print(f"Plotting predictions for {model_name} - {target_col}...")
    
    plt.figure(figsize=(12, 6))
    
    if model_name in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Classical ML models
        y_pred = model.predict(X_test)
    
    elif model_name == 'lstm':
        # LSTM model
        model.eval()
        with torch.no_grad():
            # Reshape input for LSTM [batch, sequence_length, features]
            X_test_reshaped = torch.tensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), dtype=torch.float32)
            y_pred = model(X_test_reshaped).numpy()
    
    # If we have a scaler, inverse transform the predictions and actual values
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = scaler_y.inverse_transform(y_test).flatten()
    else:
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()
    
    # Plot predictions and actual values
    plt.plot(test_dates, y_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(test_dates, y_pred, label='Predicted', color='red', alpha=0.7)
    
    plt.title(f'{model_name} - {target_col} Predictions')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/{model_name}_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{model_name}_{target_col}_predictions.png'")
    
    plt.close()

def plot_time_series_predictions(data, target_col, test_size=0.2, savefig=True):
    """
    Plot time series model predictions against actual values.
    
    Args:
        data: DataFrame with time series data
        target_col: Target column name
        test_size: Proportion of data to use for testing
        savefig: Whether to save the figure to file
    """
    print(f"Plotting time series predictions for {target_col}...")
    
    # Split data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Plot for ARIMA
    try:
        plt.figure(figsize=(12, 6))
        
        # Try to load SARIMAX model
        try:
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for test period
            exog_test = pd.get_dummies(test_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1],
                exog=exog_test
            ).predicted_mean
            
            model_name = 'SARIMAX'
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1]
            ).predicted_mean
            
            model_name = 'ARIMA'
        
        # Plot predictions and actual values
        y_true = test_data[target_col]
        plt.plot(test_data.index, y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(test_data.index, y_pred, label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'{model_name} - {target_col} Predictions')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if savefig:
            plt.savefig(f'visualizations/arima_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to 'visualizations/arima_{target_col}_predictions.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting ARIMA predictions: {e}")
    
    # Plot for Prophet
    try:
        plt.figure(figsize=(12, 6))
        
        # Load Prophet model
        import pickle
        with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare future dataframe
        future = pd.DataFrame({'ds': test_data.index})
        
        # Add weather features as regressors if they were used in training
        weather_features = [col for col in train_data.columns if col in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']]
        for feature in weather_features:
            future[feature] = test_data[feature]
        
        # Generate predictions
        forecast = model.predict(future)
        y_pred = forecast['yhat'].values
        
        # Plot predictions and actual values
        y_true = test_data[target_col]
        plt.plot(test_data.index, y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(test_data.index, y_pred, label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'Prophet - {target_col} Predictions')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if savefig:
            plt.savefig(f'visualizations/prophet_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to 'visualizations/prophet_{target_col}_predictions.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting Prophet predictions: {e}")

def plot_feature_importance(model, feature_names, model_name, target_col, savefig=True):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    if model_name not in ['random_forest', 'gradient_boosting']:
        print(f"Feature importance not available for {model_name}")
        return
    
    print(f"Plotting feature importance for {model_name} - {target_col}...")
    
    # Get feature importance from model
    importance = model.feature_importances_
    
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]
    
    # Get top 15 features
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importance, align='center')
    plt.yticks(range(top_n), top_features)
    plt.title(f'Feature Importance ({model_name} - {target_col})')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'visualizations/{model_name}_{target_col}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{model_name}_{target_col}_feature_importance.png'")
    
    plt.close()

def plot_target_distribution(data, target_col, savefig=True):
    """
    Plot the distribution of the target variable.
    
    Args:
        data: DataFrame with the target variable
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting distribution of {target_col}...")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/{target_col}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{target_col}_distribution.png'")
    
    plt.close()

def plot_correlation_heatmap(data, target_col, savefig=True):
    """
    Plot correlation heatmap between weather features and target variable.
    
    Args:
        data: DataFrame with weather features and target variable
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting correlation heatmap for {target_col}...")
    
    # Select weather features and target variable
    weather_features = [col for col in data.columns if col in [
        'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover',
        'diffuse_radiation', 'direct_radiation', 'dew_point', 'humidity'
    ]]
    
    if not weather_features:
        print("No weather features found in the data")
        return
    
    selected_cols = weather_features + [target_col]
    
    # Calculate correlation matrix
    corr = data[selected_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation between Weather Features and {target_col}')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'visualizations/{target_col}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{target_col}_correlation_heatmap.png'")
    
    plt.close()

#############################################################
# PART 5: FORECASTING FUNCTIONS
#############################################################

def forecast_renewable_energy(weather_data, forecast_horizon=24, target_col='solar_generation',
                              model_type='gradient_boosting'):
    """
    Generate renewable energy forecasts based on weather data.
    
    Args:
        weather_data: DataFrame with weather features
        forecast_horizon: Forecast horizon in hours
        target_col: Target to forecast ('solar_generation' or 'wind_generation')
        model_type: Type of model to use for forecasting
    
    Returns:
        DataFrame with forecasts
    """
    print(f"Generating {forecast_horizon}h {target_col} forecast using {model_type}...")
    
    # Load model and necessary data
    import joblib
    import json
    
    if model_type in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Load classical ML model
        model = joblib.load(f'models/{model_type}_{target_col}.pkl')
        
        # Load feature names
        with open(f'models/feature_names_{target_col}.txt', 'r') as f:
            feature_names = f.read().splitlines()
        
        # Load scalers
        scaler_X = joblib.load(f'models/scaler_X_{target_col}.pkl')
        scaler_y = joblib.load(f'models/scaler_y_{target_col}.pkl')
        
        # Prepare input features
        X = weather_data[feature_names].values
        X_scaled = scaler_X.transform(X)
        
        # Generate predictions
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred.flatten()
        })
        
    elif model_type == 'lstm':
        # Load LSTM model info
        with open(f'models/lstm_info_{target_col}.json', 'r') as f:
            model_info = json.load(f)
        
        # Initialize model with saved architecture
        input_size = model_info['input_size']
        hidden_size = model_info['hidden_size']
        num_layers = model_info['num_layers']
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Load model weights
        model.load_state_dict(torch.load(f'models/lstm_{target_col}.pt'))
        model.eval()
        
        # Load feature names
        feature_names = model_info['features']
        
        # Load scalers
        scaler_X = joblib.load(f'models/scaler_X_{target_col}.pkl')
        scaler_y = joblib.load(f'models/scaler_y_{target_col}.pkl')
        
        # Prepare input features
        X = weather_data[feature_names].values
        X_scaled = scaler_X.transform(X)
        
        # Reshape for LSTM [batch, sequence_length, features]
        X_reshaped = torch.tensor(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), dtype=torch.float32)
        
        # Generate predictions
        with torch.no_grad():
            y_pred_scaled = model(X_reshaped).numpy()
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred.flatten()
        })
    
    elif model_type == 'arima':
        try:
            # Try to load SARIMAX model
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for forecast period
            exog = pd.get_dummies(weather_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=weather_data.index[0],
                end=weather_data.index[-1],
                exog=exog
            ).predicted_mean
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=weather_data.index[0],
                end=weather_data.index[-1]
            ).predicted_mean
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred
        })
    
    elif model_type == 'prophet':
        # Load Prophet model
        import pickle
        with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare future DataFrame
        future = pd.DataFrame({'ds': weather_data.index})
        
        # Add weather features as regressors if they were used in training
        weather_features = [col for col in weather_data.columns if col in [
            'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover'
        ]]
        
        for feature in weather_features:
            future[feature] = weather_data[feature].values
        
        # Generate predictions
        forecast_df = model.predict(future)
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': forecast_df['yhat'].values
        })
    
    # Set timestamp as index
    forecast = forecast.set_index('timestamp')
    
    # Save forecast
    forecast.to_csv(f'results/{model_type}_{target_col}_forecast.csv')
    print(f"Forecast saved to 'results/{model_type}_{target_col}_forecast.csv'")
    
    return forecast

def plot_forecast(forecast, target_col, model_type, savefig=True):
    """
    Plot forecast.
    
    Args:
        forecast: DataFrame with forecasts
        target_col: Target column name
        model_type: Type of model used for forecasting
        savefig: Whether to save the figure to file
    """
    print(f"Plotting {model_type} forecast for {target_col}...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot forecast
    plt.plot(forecast.index, forecast[f'{target_col}_forecast'], color='red', alpha=0.7)
    
    plt.title(f'{model_type} - {target_col} Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    
    # Add confidence intervals for Prophet
    if model_type == 'prophet' and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        plt.fill_between(
            forecast.index,
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color='red',
            alpha=0.1
        )
    
    if savefig:
        plt.savefig(f'visualizations/{model_type}_{target_col}_forecast.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{model_type}_{target_col}_forecast.png'")
    
    plt.close()

def create_ensemble_forecast(forecasts, target_col, weights=None):
    """
    Create ensemble forecast by combining multiple model forecasts.
    
    Args:
        forecasts: Dictionary of DataFrames with forecasts from different models
        target_col: Target column name
        weights: Dictionary of weights for each model (optional)
    
    Returns:
        DataFrame with ensemble forecast
    """
    print(f"Creating ensemble forecast for {target_col}...")
    
    # Get list of models
    models = list(forecasts.keys())
    
    # Create DataFrame for ensemble forecast
    ensemble = pd.DataFrame(index=forecasts[models[0]].index)
    
    # Add individual model forecasts
    for model in models:
        ensemble[f'{model}_forecast'] = forecasts[model][f'{target_col}_forecast']
    
    # Calculate ensemble forecast
    if weights is None:
        # Simple average
        ensemble[f'{target_col}_ensemble'] = ensemble.mean(axis=1)
    else:
        # Weighted average
        weighted_sum = 0
        total_weight = 0
        
        for model in models:
            if model in weights:
                weighted_sum += ensemble[f'{model}_forecast'] * weights[model]
                total_weight += weights[model]
        
        ensemble[f'{target_col}_ensemble'] = weighted_sum / total_weight
    
    # Save ensemble forecast
    ensemble.to_csv(f'results/ensemble_{target_col}_forecast.csv')
    print(f"Ensemble forecast saved to 'results/ensemble_{target_col}_forecast.csv'")
    
    return ensemble

def plot_ensemble_forecast(ensemble, target_col, savefig=True):
    """
    Plot ensemble forecast.
    
    Args:
        ensemble: DataFrame with ensemble forecast
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting ensemble forecast for {target_col}...")
    
    plt.figure(figsize=(12, 6))
    
    # Get list of models
    models = [col.split('_')[0] for col in ensemble.columns if col.endswith('_forecast')]
    
    # Plot individual model forecasts
    for model in models:
        plt.plot(ensemble.index, ensemble[f'{model}_forecast'], alpha=0.3, label=f'{model}')
    
    # Plot ensemble forecast
    plt.plot(ensemble.index, ensemble[f'{target_col}_ensemble'], color='black', linewidth=2, label='Ensemble')
    
    plt.title(f'Ensemble Forecast - {target_col}')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/ensemble_{target_col}_forecast.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/ensemble_{target_col}_forecast.png'")
    
    plt.close()



#############################################################
# PART 4: MODEL EVALUATION
#############################################################

def evaluate_model(model, X_test, y_test, model_name, target_col, scaler_y=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        target_col: Target column name
        scaler_y: Scaler used to transform the target
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating {model_name} for {target_col}...")
    
    if model_name in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Classical ML models
        y_pred = model.predict(X_test)
    
    elif model_name == 'lstm':
        # LSTM model
        model.eval()
        with torch.no_grad():
            # Reshape input for LSTM [batch, sequence_length, features]
            X_test_reshaped = torch.tensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), dtype=torch.float32)
            y_pred = model(X_test_reshaped).numpy()
    
    # If we have a scaler, inverse transform the predictions and actual values
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model': model_name,
        'target': target_col,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    # Save metrics
    import json
    with open(f'results/{model_name}_{target_col}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def evaluate_time_series_models(data, target_col, test_size=0.2):
    """
    Evaluate time series models (ARIMA, Prophet).
    
    Args:
        data: DataFrame with time series data
        target_col: Target column name
        test_size: Proportion of data to use for testing
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating time series models for {target_col}...")
    
    # Split data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    results = {}
    
    # Evaluate ARIMA
    try:
        print("Evaluating ARIMA model...")
        
        # Try to load SARIMAX model
        try:
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for test period
            exog_test = pd.get_dummies(test_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1],
                exog=exog_test
            ).predicted_mean
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1]
            ).predicted_mean
        
        # Calculate metrics
        y_true = test_data[target_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'model': 'arima',
            'target': target_col,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save metrics
        import json
        with open(f'results/arima_{target_col}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        results['arima'] = metrics
        
    except Exception as e:
        print(f"Error evaluating ARIMA model: {e}")
    
    # Evaluate Prophet or Exponential Smoothing
    try:
        if PROPHET_AVAILABLE:
            print("Evaluating Prophet model...")
            
            # Load Prophet model
            import pickle
            with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare future dataframe
            future = pd.DataFrame({'ds': test_data.index})
            
            # Add weather features as regressors if they were used in training
            weather_features = [col for col in train_data.columns if col in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']]
            for feature in weather_features:
                future[feature] = test_data[feature]
            
            # Generate predictions
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
            
        else:
            print("Evaluating Exponential Smoothing model...")
            
            # Try to load exponential smoothing model
            import pickle
            try:
                with open(f'models/exp_smoothing_{target_col}.pkl', 'rb') as f:
                    model = pickle.load(f)
            except:
                with open(f'models/simple_exp_smoothing_{target_col}.pkl', 'rb') as f:
                    model = pickle.load(f)
            
            # Generate predictions
            y_pred = model.forecast(len(test_data))
        
        # Calculate metrics
        y_true = test_data[target_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        model_name = 'prophet' if PROPHET_AVAILABLE else 'exp_smoothing'
        
        metrics = {
            'model': model_name,
            'target': target_col,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save metrics
        import json
        with open(f'results/{model_name}_{target_col}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        results[model_name] = metrics
        
    except Exception as e:
        print(f"Error evaluating Prophet/Exponential Smoothing model: {e}")
    
    return results

def plot_predictions(model, X_test, y_test, test_dates, model_name, target_col, scaler_y=None, savefig=True):
    """
    Plot model predictions against actual values.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        test_dates: Test dates (for x-axis)
        model_name: Name of the model
        target_col: Target column name
        scaler_y: Scaler used to transform the target
        savefig: Whether to save the figure to file
    """
    print(f"Plotting predictions for {model_name} - {target_col}...")
    
    plt.figure(figsize=(12, 6))
    
    if model_name in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Classical ML models
        y_pred = model.predict(X_test)
    
    elif model_name == 'lstm':
        # LSTM model
        model.eval()
        with torch.no_grad():
            # Reshape input for LSTM [batch, sequence_length, features]
            X_test_reshaped = torch.tensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), dtype=torch.float32)
            y_pred = model(X_test_reshaped).numpy()
    
    # If we have a scaler, inverse transform the predictions and actual values
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = scaler_y.inverse_transform(y_test).flatten()
    else:
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()
    
    # Plot predictions and actual values
    plt.plot(test_dates, y_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(test_dates, y_pred, label='Predicted', color='red', alpha=0.7)
    
    plt.title(f'{model_name} - {target_col} Predictions')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/{model_name}_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{model_name}_{target_col}_predictions.png'")
    
    plt.close()

def plot_time_series_predictions(data, target_col, test_size=0.2, savefig=True):
    """
    Plot time series model predictions against actual values.
    
    Args:
        data: DataFrame with time series data
        target_col: Target column name
        test_size: Proportion of data to use for testing
        savefig: Whether to save the figure to file
    """
    print(f"Plotting time series predictions for {target_col}...")
    
    # Split data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Plot for ARIMA
    try:
        plt.figure(figsize=(12, 6))
        
        # Try to load SARIMAX model
        try:
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for test period
            exog_test = pd.get_dummies(test_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1],
                exog=exog_test
            ).predicted_mean
            
            model_name = 'SARIMAX'
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1]
            ).predicted_mean
            
            model_name = 'ARIMA'
        
        # Plot predictions and actual values
        y_true = test_data[target_col]
        plt.plot(test_data.index, y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(test_data.index, y_pred, label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'{model_name} - {target_col} Predictions')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if savefig:
            plt.savefig(f'visualizations/arima_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to 'visualizations/arima_{target_col}_predictions.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting ARIMA predictions: {e}")
    
    # Plot for Prophet or Exponential Smoothing
    try:
        plt.figure(figsize=(12, 6))
        
        if PROPHET_AVAILABLE:
            # Load Prophet model
            import pickle
            with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare future dataframe
            future = pd.DataFrame({'ds': test_data.index})
            
            # Add weather features as regressors if they were used in training
            weather_features = [col for col in train_data.columns if col in ['temperature', 'wind_speed', 'solar_radiation', 'cloud_cover']]
            for feature in weather_features:
                future[feature] = test_data[feature]
            
            # Generate predictions
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
            
            model_name = 'Prophet'
            
        else:
            # Load exponential smoothing model
            import pickle
            try:
                with open(f'models/exp_smoothing_{target_col}.pkl', 'rb') as f:
                    model = pickle.load(f)
            except:
                with open(f'models/simple_exp_smoothing_{target_col}.pkl', 'rb') as f:
                    model = pickle.load(f)
            
            # Generate predictions
            y_pred = model.forecast(len(test_data))
            
            model_name = 'Exponential Smoothing'
        
        # Plot predictions and actual values
        y_true = test_data[target_col]
        plt.plot(test_data.index, y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(test_data.index, y_pred, label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'{model_name} - {target_col} Predictions')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        model_file_name = 'prophet' if PROPHET_AVAILABLE else 'exp_smoothing'
        
        if savefig:
            plt.savefig(f'visualizations/{model_file_name}_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to 'visualizations/{model_file_name}_{target_col}_predictions.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting Prophet/Exponential Smoothing predictions: {e}")

def plot_feature_importance(model, feature_names, model_name, target_col, savefig=True):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    if model_name not in ['random_forest', 'gradient_boosting']:
        print(f"Feature importance not available for {model_name}")
        return
    
    print(f"Plotting feature importance for {model_name} - {target_col}...")
    
    # Get feature importance from model
    importance = model.feature_importances_
    
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]
    
    # Get top 15 features
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importance, align='center')
    plt.yticks(range(top_n), top_features)
    plt.title(f'Feature Importance ({model_name} - {target_col})')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'visualizations/{model_name}_{target_col}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{model_name}_{target_col}_feature_importance.png'")
    
    plt.close()

def plot_target_distribution(data, target_col, savefig=True):
    """
    Plot the distribution of the target variable.
    
    Args:
        data: DataFrame with the target variable
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting distribution of {target_col}...")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/{target_col}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{target_col}_distribution.png'")
    
    plt.close()

def plot_correlation_heatmap(data, target_col, savefig=True):
    """
    Plot correlation heatmap between weather features and target variable.
    
    Args:
        data: DataFrame with weather features and target variable
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting correlation heatmap for {target_col}...")
    
    # Select weather features and target variable
    weather_features = [col for col in data.columns if col in [
        'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover',
        'diffuse_radiation', 'direct_radiation', 'dew_point', 'humidity'
    ]]
    
    if not weather_features:
        print("No weather features found in the data")
        return
    
    selected_cols = weather_features + [target_col]
    
    # Calculate correlation matrix
    corr = data[selected_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation between Weather Features and {target_col}')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'visualizations/{target_col}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/{target_col}_correlation_heatmap.png'")
    
    plt.close()

#############################################################
# PART 5: FORECASTING FUNCTIONS
#############################################################

def forecast_renewable_energy(weather_data, forecast_horizon=24, target_col='solar_generation',
                              model_type='gradient_boosting'):
    """
    Generate renewable energy forecasts based on weather data.
    
    Args:
        weather_data: DataFrame with weather features
        forecast_horizon: Forecast horizon in hours
        target_col: Target to forecast ('solar_generation' or 'wind_generation')
        model_type: Type of model to use for forecasting
    
    Returns:
        DataFrame with forecasts
    """
    print(f"Generating {forecast_horizon}h {target_col} forecast using {model_type}...")
    
    # Load model and necessary data
    import joblib
    import json
    
    if model_type in ['linear_regression', 'random_forest', 'gradient_boosting']:
        # Load classical ML model
        model = joblib.load(f'models/{model_type}_{target_col}.pkl')
        
        # Load feature names
        with open(f'models/feature_names_{target_col}.txt', 'r') as f:
            feature_names = f.read().splitlines()
        
        # Load scalers
        scaler_X = joblib.load(f'models/scaler_X_{target_col}.pkl')
        scaler_y = joblib.load(f'models/scaler_y_{target_col}.pkl')
        
        # Prepare input features
        X = weather_data[feature_names].values
        X_scaled = scaler_X.transform(X)
        
        # Generate predictions
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred.flatten()
        })
        
    elif model_type == 'lstm':
        # Load LSTM model info
        with open(f'models/lstm_info_{target_col}.json', 'r') as f:
            model_info = json.load(f)
        
        # Initialize model with saved architecture
        input_size = model_info['input_size']
        hidden_size = model_info['hidden_size']
        num_layers = model_info['num_layers']
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Load model weights
        model.load_state_dict(torch.load(f'models/lstm_{target_col}.pt'))
        model.eval()
        
        # Load feature names
        feature_names = model_info['features']
        
        # Load scalers
        scaler_X = joblib.load(f'models/scaler_X_{target_col}.pkl')
        scaler_y = joblib.load(f'models/scaler_y_{target_col}.pkl')
        
        # Prepare input features
        X = weather_data[feature_names].values
        X_scaled = scaler_X.transform(X)
        
        # Reshape for LSTM [batch, sequence_length, features]
        X_reshaped = torch.tensor(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), dtype=torch.float32)
        
        # Generate predictions
        with torch.no_grad():
            y_pred_scaled = model(X_reshaped).numpy()
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred.flatten()
        })
    
    elif model_type == 'arima':
        try:
            # Try to load SARIMAX model
            model = sm.load(f'models/sarimax_{target_col}.pkl')
            
            # Create exogenous variables for forecast period
            exog = pd.get_dummies(weather_data.index.hour, prefix='hour')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=weather_data.index[0],
                end=weather_data.index[-1],
                exog=exog
            ).predicted_mean
            
        except Exception as e:
            print(f"Error loading SARIMAX model: {e}")
            print("Trying to load simpler ARIMA model...")
            
            # Try to load simpler ARIMA model
            model = sm.load(f'models/arima_{target_col}.pkl')
            
            # Generate predictions
            y_pred = model.get_prediction(
                start=weather_data.index[0],
                end=weather_data.index[-1]
            ).predicted_mean
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': y_pred
        })

    elif model_type == 'prophet':
        # Load Prophet model
        import pickle
        with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare future DataFrame
        future = pd.DataFrame({'ds': weather_data.index})
        
        # Add weather features as regressors if they were used in training
        weather_features = [col for col in weather_data.columns if col in [
            'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover'
        ]]
        
        for feature in weather_features:
            future[feature] = weather_data[feature].values
        
        # Generate predictions
        forecast_df = model.predict(future)
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            'timestamp': weather_data.index,
            f'{target_col}_forecast': forecast_df['yhat'].values
        })

    # Set timestamp as index
    forecast = forecast.set_index('timestamp')

    # Save forecast
    forecast.to_csv(f'results/{model_type}_{target_col}_forecast.csv')
    print(f"Forecast saved to 'results/{model_type}_{target_col}_forecast.csv'")
    return forecast



############################################################
# PART 6: MAIN FUNCTION
#############################################################

def main():
    """
    Main function to run the entire pipeline.
    """
    print("==========================================")
    print("Renewable Energy Output Forecasting Project")
    print("==========================================")
    
    # Step 1: Data Acquisition
    print("\n--- Step 1: Data Acquisition ---\n")
    
    weather_data = download_weather_data()
    power_data = download_power_generation_data()
    
    # Step 2: Data Preprocessing
    print("\n--- Step 2: Data Preprocessing ---\n")
    
    weather_processed = preprocess_weather_data(weather_data)
    power_processed = preprocess_power_data(power_data)
    merged_data = merge_weather_and_power_data(weather_processed, power_processed)
    feature_engineered_data = create_lagged_features(merged_data)
    
    # Step 3: Exploratory Data Analysis
    print("\n--- Step 3: Exploratory Data Analysis ---\n")
    
    # Plot target distributions
    plot_target_distribution(feature_engineered_data, 'solar_generation')
    plot_target_distribution(feature_engineered_data, 'wind_generation')
    
    # Plot correlation heatmaps
    plot_correlation_heatmap(feature_engineered_data, 'solar_generation')
    plot_correlation_heatmap(feature_engineered_data, 'wind_generation')
    
    # Step 4: Model Building and Evaluation for Solar Generation
    print("\n--- Step 4: Model Building and Evaluation for Solar Generation ---\n")
    
    # Prepare train/test data for solar generation
    X_train_solar, X_test_solar, y_train_solar, y_test_solar, feature_names_solar, scaler_X_solar, scaler_y_solar = prepare_train_test_data(
        feature_engineered_data, 'solar_generation'
    )
    
    # Build and evaluate classical models for solar generation
    classical_models_solar = build_classical_models(X_train_solar, y_train_solar, 'solar_generation')
    
    for name, model in classical_models_solar.items():
        metrics = evaluate_model(model, X_test_solar, y_test_solar, name, 'solar_generation', scaler_y_solar)
        plot_predictions(model, X_test_solar, y_test_solar, feature_engineered_data.index[-len(X_test_solar):], name, 'solar_generation', scaler_y_solar)
        
        if name in ['random_forest', 'gradient_boosting']:
            plot_feature_importance(model, feature_names_solar, name, 'solar_generation')
    
    # Build and evaluate LSTM model for solar generation
    lstm_model_solar, _ = build_lstm_model(X_train_solar, y_train_solar, feature_names_solar, 'solar_generation')
    evaluate_model(lstm_model_solar, X_test_solar, y_test_solar, 'lstm', 'solar_generation', scaler_y_solar)
    plot_predictions(lstm_model_solar, X_test_solar, y_test_solar, feature_engineered_data.index[-len(X_test_solar):], 'lstm', 'solar_generation', scaler_y_solar)
    
    # Build and evaluate time series models for solar generation
    build_arima_model(feature_engineered_data, 'solar_generation')
    build_prophet_model(feature_engineered_data, 'solar_generation')
    evaluate_time_series_models(feature_engineered_data, 'solar_generation')
    plot_time_series_predictions(feature_engineered_data, 'solar_generation')
    
    # Step 5: Model Building and Evaluation for Wind Generation
    print("\n--- Step 5: Model Building and Evaluation for Wind Generation ---\n")
    
    # Prepare train/test data for wind generation
    X_train_wind, X_test_wind, y_train_wind, y_test_wind, feature_names_wind, scaler_X_wind, scaler_y_wind = prepare_train_test_data(
        feature_engineered_data, 'wind_generation'
    )
    
    # Build and evaluate classical models for wind generation
    classical_models_wind = build_classical_models(X_train_wind, y_train_wind, 'wind_generation')
    
    for name, model in classical_models_wind.items():
        metrics = evaluate_model(model, X_test_wind, y_test_wind, name, 'wind_generation', scaler_y_wind)
        plot_predictions(model, X_test_wind, y_test_wind, feature_engineered_data.index[-len(X_test_wind):], name, 'wind_generation', scaler_y_wind)
        
        if name in ['random_forest', 'gradient_boosting']:
            plot_feature_importance(model, feature_names_wind, name, 'wind_generation')
    
    # Build and evaluate LSTM model for wind generation
    lstm_model_wind, _ = build_lstm_model(X_train_wind, y_train_wind, feature_names_wind, 'wind_generation')
    evaluate_model(lstm_model_wind, X_test_wind, y_test_wind, 'lstm', 'wind_generation', scaler_y_wind)
    plot_predictions(lstm_model_wind, X_test_wind, y_test_wind, feature_engineered_data.index[-len(X_test_wind):], 'lstm', 'wind_generation', scaler_y_wind)
    
    # Build and evaluate time series models for wind generation
    build_arima_model(feature_engineered_data, 'wind_generation')
    build_prophet_model(feature_engineered_data, 'wind_generation')
    evaluate_time_series_models(feature_engineered_data, 'wind_generation')
    plot_time_series_predictions(feature_engineered_data, 'wind_generation')
    
    # Step 6: Forecasting
    print("\n--- Step 6: Forecasting ---\n")
    
    # Create a future weather data frame for forecasting (for demonstration, we'll use the last 24 hours of our data)
    future_weather = weather_processed.iloc[-24:].copy()
    
    # Generate forecasts for solar generation
    forecasts_solar = {}
    for model_type in ['linear_regression', 'random_forest', 'gradient_boosting', 'lstm', 'arima', 'prophet']:
        try:
            forecast = forecast_renewable_energy(future_weather, 24, 'solar_generation', model_type)
            forecasts_solar[model_type] = forecast
            plot_forecast(forecast, 'solar_generation', model_type)
        except Exception as e:
            print(f"Error generating forecast with {model_type} for solar_generation: {e}")
    
    # Create ensemble forecast for solar generation
    if forecasts_solar:
        ensemble_solar = create_ensemble_forecast(forecasts_solar, 'solar_generation')
        plot_ensemble_forecast(ensemble_solar, 'solar_generation')
    
    # Generate forecasts for wind generation
    forecasts_wind = {}
    for model_type in ['linear_regression', 'random_forest', 'gradient_boosting', 'lstm', 'arima', 'prophet']:
        try:
            forecast = forecast_renewable_energy(future_weather, 24, 'wind_generation', model_type)
            forecasts_wind[model_type] = forecast
            plot_forecast(forecast, 'wind_generation', model_type)
        except Exception as e:
            print(f"Error generating forecast with {model_type} for wind_generation: {e}")
    
    # Create ensemble forecast for wind generation
    if forecasts_wind:
        ensemble_wind = create_ensemble_forecast(forecasts_wind, 'wind_generation')
        plot_ensemble_forecast(ensemble_wind, 'wind_generation')
    
    print("\n==========================================")
    print("Project complete! Results saved in 'results/' directory.")
    print("Visualizations saved in 'visualizations/' directory.")
    print("==========================================")

if __name__ == "__main__":
    def plot_forecast(forecast, target_col, model_type, savefig=True):
        """
        Plot forecast.
        
        Args:
            forecast: DataFrame with forecasts
            target_col: Target column name
            model_type: Type of model used for forecasting
            savefig: Whether to save the figure to file
        """
        print(f"Plotting {model_type} forecast for {target_col}...")
        
        plt.figure(figsize=(12, 6))
        
        # Plot forecast
        plt.plot(forecast.index, forecast[f'{target_col}_forecast'], color='red', alpha=0.7)
        
        plt.title(f'{model_type} - {target_col} Forecast')
        plt.xlabel('Timestamp')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        
        # Add confidence intervals for Prophet
        if model_type == 'prophet' and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            plt.fill_between(
                forecast.index,
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                color='red',
                alpha=0.1
            )
        
        if savefig:
            plt.savefig(f'visualizations/{model_type}_{target_col}_forecast.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to 'visualizations/{model_type}_{target_col}_forecast.png'")
        
        plt.close()

def create_ensemble_forecast(forecasts, target_col, weights=None):
    """
    Create ensemble forecast by combining multiple model forecasts.
    
    Args:
        forecasts: Dictionary of DataFrames with forecasts from different models
        target_col: Target column name
        weights: Dictionary of weights for each model (optional)
    
    Returns:
        DataFrame with ensemble forecast
    """
    print(f"Creating ensemble forecast for {target_col}...")
    
    # Get list of models
    models = list(forecasts.keys())
    
    # Create DataFrame for ensemble forecast
    ensemble = pd.DataFrame(index=forecasts[models[0]].index)
    
    # Add individual model forecasts
    for model in models:
        ensemble[f'{model}_forecast'] = forecasts[model][f'{target_col}_forecast']
    
    # Calculate ensemble forecast
    if weights is None:
        # Simple average
        ensemble[f'{target_col}_ensemble'] = ensemble.mean(axis=1)
    else:
        # Weighted average
        weighted_sum = 0
        total_weight = 0
        
        for model in models:
            if model in weights:
                weighted_sum += ensemble[f'{model}_forecast'] * weights[model]
                total_weight += weights[model]
        
        ensemble[f'{target_col}_ensemble'] = weighted_sum / total_weight
    
    # Save ensemble forecast
    ensemble.to_csv(f'results/ensemble_{target_col}_forecast.csv')
    print(f"Ensemble forecast saved to 'results/ensemble_{target_col}_forecast.csv'")
    
    return ensemble

def plot_ensemble_forecast(ensemble, target_col, savefig=True):
    """
    Plot ensemble forecast.
    
    Args:
        ensemble: DataFrame with ensemble forecast
        target_col: Target column name
        savefig: Whether to save the figure to file
    """
    print(f"Plotting ensemble forecast for {target_col}...")
    
    plt.figure(figsize=(12, 6))
    
    # Get list of models
    models = [col.split('_')[0] for col in ensemble.columns if col.endswith('_forecast')]
    
    # Plot individual model forecasts
    for model in models:
        plt.plot(ensemble.index, ensemble[f'{model}_forecast'], alpha=0.3, label=f'{model}')
    
    # Plot ensemble forecast
    plt.plot(ensemble.index, ensemble[f'{target_col}_ensemble'], color='black', linewidth=2, label='Ensemble')
    
    plt.title(f'Ensemble Forecast - {target_col}')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig(f'visualizations/ensemble_{target_col}_forecast.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'visualizations/ensemble_{target_col}_forecast.png'")
    
    plt.close()