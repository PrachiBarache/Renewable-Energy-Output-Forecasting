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
import warnings
warnings.filterwarnings("ignore")
# Add at the top of the file
import os
# Limit memory usage for TensorFlow (which might be loaded by Prophet)
os.environ['TF_MEMORY_ALLOCATION'] = '2048MB'
# Set random seed for reproducibility
np.random.seed(42)

# Create directories for saving data and models
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

#############################################################
# PART 1: DATA ACQUISITION
#############################################################

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

def load_weather_data(file_path=None):
    """
    Load locally downloaded weather data or create synthetic data if not available.
    """
    print("Loading weather data...")
    
    # First check if we have a synthetic dataset already
    synthetic_path = 'data/raw/weather_data_synthetic.csv'
    if os.path.exists(synthetic_path):
        print(f"Loading existing synthetic weather data from: {synthetic_path}")
        return pd.read_csv(synthetic_path)
    
    # If a file path is provided, try to load it
    if file_path and os.path.exists(file_path):
        try:
            # Check if the file contains NREL headers (they often have metadata in first rows)
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # If it's NREL NSRDB data with metadata headers
                if 'PSM' in first_line or 'NSRDB' in first_line:
                    weather_data = pd.read_csv(file_path, skiprows=2)
                else:
                    weather_data = pd.read_csv(file_path)
            
            print(f"Weather data loaded successfully with {len(weather_data)} rows and {len(weather_data.columns)} columns")
            print(f"Columns: {weather_data.columns.tolist()}")
            
            return weather_data
        except Exception as e:
            print(f"Error loading weather data: {e}")
    
    # If we get here, either no file path was provided or loading failed
    print("Creating synthetic weather data...")
    return create_synthetic_weather_data()

def load_power_data(file_path=None):
    """
    Load locally downloaded power generation data or create synthetic data if not available.
    """
    print("Loading power generation data...")
    
    # First check if we have a synthetic dataset already
    synthetic_path = 'data/raw/power_generation_synthetic.csv'
    if os.path.exists(synthetic_path):
        print(f"Loading existing synthetic power data from: {synthetic_path}")
        return pd.read_csv(synthetic_path)
    
    # If a file path is provided, try to load it
    if file_path and os.path.exists(file_path):
        try:
            power_data = pd.read_csv(file_path)
            
            print(f"Power generation data loaded successfully with {len(power_data)} rows and {len(power_data.columns)} columns")
            print(f"Columns: {power_data.columns.tolist()}")
            
            return power_data
        except Exception as e:
            print(f"Error loading power generation data: {e}")
    
    # If we get here, either no file path was provided or loading failed
    print("Creating synthetic power generation data...")
    return create_synthetic_power_data()

#############################################################
# PART 2: DATA PREPROCESSING
#############################################################

def preprocess_weather_data(weather_data):
    """
    Preprocess the weather data.
    """
    print("Preprocessing weather data...")
    
    # Make a copy to avoid modifying the original
    weather_data = weather_data.copy()
    
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
    
    # If timestamp exists as a column, ensure it's in datetime format
    if 'timestamp' in weather_data.columns:
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        weather_data = weather_data.set_index('timestamp')
    else:
        # If no timestamp column, try to create one from year, month, day, hour if those exist
        if all(col in weather_data.columns for col in ['year', 'month', 'day', 'hour']):
            weather_data['timestamp'] = pd.to_datetime(
                weather_data[['year', 'month', 'day', 'hour']]
            )
            weather_data = weather_data.set_index('timestamp')
            
    # Select relevant features
    relevant_features = [col for col in weather_data.columns if col in [
        'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover',
        'diffuse_radiation', 'direct_radiation', 'dew_point', 'humidity',
        'month', 'hour', 'day_of_year'
    ]]
    
    weather_processed = weather_data[relevant_features].copy()
    
    # Handle missing values
    weather_processed = weather_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Add engineered features if they don't exist
    if 'hour' not in weather_processed.columns:
        weather_processed['hour'] = weather_processed.index.hour
    if 'day_of_year' not in weather_processed.columns:
        weather_processed['day_of_year'] = weather_processed.index.dayofyear
    if 'month' not in weather_processed.columns:
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
    
    # Make a copy to avoid modifying the original
    power_data = power_data.copy()
    
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
    
    # Memory safety - check if we need to reduce dataset size
    max_samples = 10000  # Adjust this based on available memory
    if len(df_filtered) > max_samples:
        print(f"Dataset is large ({len(df_filtered)} samples). Reducing to {max_samples} samples.")
        # Only sample if the dataset is larger than max_samples
        df_filtered = df_filtered.sample(max_samples, random_state=42)
    
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
    
    Args:
        X_train: Training features
        y_train: Training targets
        target_col: Target column name
    
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
    Build and train Prophet model.
    
    Args:
        data: DataFrame with time series data
        target_col: Target column name
    
    Returns:
        Trained model
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

# ISSUE 1: Fix for the forecast_renewable_energy function
def forecast_renewable_energy(weather_data, forecast_horizon=24, target_col='solar_generation',
                              model_type='gradient_boosting'):
    """
    Generate renewable energy forecasts based on weather data.
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

# ISSUE 2: Properly implementing create_ensemble_forecast
def create_ensemble_forecast(forecasts, target_col, weights=None):
    """
    Create ensemble forecast by combining multiple model forecasts.
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

# ISSUE 3: Properly implementing plot_ensemble_forecast
def plot_ensemble_forecast(ensemble, target_col, savefig=True):
    """
    Plot ensemble forecast.
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

def plot_target_distribution(data, target_col):
    """Plot distribution of target variable."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.savefig(f'visualizations/{target_col}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/{target_col}_distribution.png'")

def plot_correlation_heatmap(data, target_col):
    """Plot correlation heatmap for features with target."""
    plt.figure(figsize=(12, 10))
    # Select numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Calculate correlation matrix
    corr = data[numeric_cols].corr()
    # Plot heatmap
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title(f'Feature Correlation Heatmap for {target_col}')
    plt.savefig(f'visualizations/{target_col}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/{target_col}_correlation_heatmap.png'")

def evaluate_model(model, X_test, y_test, model_name, target_col, scaler_y=None):
    """Evaluate model performance on test data."""
    # Make predictions
    y_pred = model.predict(X_test)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Inverse transform if scaler is provided
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test_inv = scaler_y.inverse_transform(y_test)
    else:
        y_test_inv = y_test
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_inv, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    r2 = r2_score(y_test_inv, y_pred)
    
    # Print metrics
    print(f"\n{model_name.upper()} Model Evaluation for {target_col}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Save metrics to file
    with open(f'results/{model_name}_{target_col}_metrics.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R²: {r2:.4f}\n")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictions(model, X_test, y_test, timestamps, model_name, target_col, scaler_y=None):
    """Plot model predictions against actual values."""
    # Make predictions
    y_pred = model.predict(X_test)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Inverse transform if scaler is provided
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test_inv = scaler_y.inverse_transform(y_test)
    else:
        y_test_inv = y_test
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y_test_inv, label='Actual', color='blue', alpha=0.7)
    plt.plot(timestamps, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'{model_name.upper()} Model - {target_col} Predictions')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{model_name}_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/{model_name}_{target_col}_predictions.png'")

def plot_feature_importance(model, feature_names, model_name, target_col):
    """Plot feature importance for tree-based models."""
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices[:15])), importances[indices[:15]], align='center')
    plt.yticks(range(len(indices[:15])), [feature_names[i] for i in indices[:15]])
    plt.title(f'{model_name.upper()} - Top 15 Feature Importances for {target_col}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_{target_col}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/{model_name}_{target_col}_feature_importance.png'")

def evaluate_time_series_models(data, target_col):
    """Evaluate time series models."""
    # Split data for validation
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Evaluate ARIMA model
    try:
        # Load SARIMAX model
        model = sm.load(f'models/sarimax_{target_col}.pkl')
        
        # Create exogenous variables
        exog = pd.get_dummies(test_data.index.hour, prefix='hour')
        
        # Generate predictions
        predictions = model.get_prediction(
            start=test_data.index[0],
            end=test_data.index[-1],
            exog=exog
        ).predicted_mean
        
        # Calculate metrics
        mae = mean_absolute_error(test_data[target_col], predictions)
        rmse = np.sqrt(mean_squared_error(test_data[target_col], predictions))
        r2 = r2_score(test_data[target_col], predictions)
        
        # Print metrics
        print(f"\nARIMA Model Evaluation for {target_col}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Save metrics to file
        with open(f'results/arima_{target_col}_metrics.txt', 'w') as f:
            f.write(f"Model: ARIMA\n")
            f.write(f"Target: {target_col}\n")
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write(f"R²: {r2:.4f}\n")
    
    except Exception as e:
        print(f"Error evaluating ARIMA model: {e}")
    
    # Evaluate Prophet model if available
    if PROPHET_AVAILABLE:
        try:
            # Load Prophet model
            import pickle
            with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare future DataFrame
            future = pd.DataFrame({'ds': test_data.index})
            
            # Add weather features as regressors if they were used in training
            weather_features = [col for col in test_data.columns if col in [
                'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover'
            ]]
            
            for feature in weather_features:
                future[feature] = test_data[feature].values
            
            # Generate predictions
            forecast = model.predict(future)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data[target_col], forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(test_data[target_col], forecast['yhat']))
            r2 = r2_score(test_data[target_col], forecast['yhat'])
            
            # Print metrics
            print(f"\nProphet Model Evaluation for {target_col}:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R²: {r2:.4f}")
            
            # Save metrics to file
            with open(f'results/prophet_{target_col}_metrics.txt', 'w') as f:
                f.write(f"Model: Prophet\n")
                f.write(f"Target: {target_col}\n")
                f.write(f"MAE: {mae:.2f}\n")
                f.write(f"RMSE: {rmse:.2f}\n")
                f.write(f"R²: {r2:.4f}\n")
        
        except Exception as e:
            print(f"Error evaluating Prophet model: {e}")

def plot_time_series_predictions(data, target_col):
    """Plot time series model predictions."""
    # Split data for validation
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data[target_col], label='Actual', color='blue', alpha=0.7)
    
    # Plot ARIMA predictions
    try:
        # Load SARIMAX model
        model = sm.load(f'models/sarimax_{target_col}.pkl')
        
        # Create exogenous variables
        exog = pd.get_dummies(test_data.index.hour, prefix='hour')
        
        # Generate predictions
        predictions = model.get_prediction(
            start=test_data.index[0],
            end=test_data.index[-1],
            exog=exog
        ).predicted_mean
        
        plt.plot(test_data.index, predictions, label='ARIMA', color='red', alpha=0.7)
    except Exception as e:
        print(f"Error plotting ARIMA predictions: {e}")
    
    # Plot Prophet predictions if available
    if PROPHET_AVAILABLE:
        try:
            # Load Prophet model
            import pickle
            with open(f'models/prophet_{target_col}.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare future DataFrame
            future = pd.DataFrame({'ds': test_data.index})
            
            # Add weather features as regressors if they were used in training
            weather_features = [col for col in test_data.columns if col in [
                'temperature', 'wind_speed', 'solar_radiation', 'cloud_cover'
            ]]
            
            for feature in weather_features:
                future[feature] = test_data[feature].values
            
            # Generate predictions
            forecast = model.predict(future)
            
            plt.plot(test_data.index, forecast['yhat'], label='Prophet', color='green', alpha=0.7)
        except Exception as e:
            print(f"Error plotting Prophet predictions: {e}")
    
    plt.title(f'Time Series Models - {target_col} Predictions')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/time_series_{target_col}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/time_series_{target_col}_predictions.png'")

def plot_forecast(forecast, target_col, model_type):
    """Plot forecast."""
    plt.figure(figsize=(12, 6))
    plt.plot(forecast.index, forecast[f'{target_col}_forecast'], label=f'{model_type} Forecast')
    plt.title(f'{model_type.upper()} - {target_col} Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'visualizations/{model_type}_{target_col}_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to 'visualizations/{model_type}_{target_col}_forecast.png'")

# ISSUE 4: Main execution block
if __name__ == "__main__":
    # Wrap all steps in try-except blocks
    try:
        print("==========================================")
        print("Renewable Energy Output Forecasting Project")
        print("==========================================")
        
        # Step 1: Data Acquisition
        print("\n--- Step 1: Data Acquisition ---\n")
        
        weather_data = load_weather_data()
        power_data = load_power_data()
        
        # Step 2: Data Preprocessing
        print("\n--- Step 2: Data Preprocessing ---\n")
        
        weather_processed = preprocess_weather_data(weather_data)
        power_processed = preprocess_power_data(power_data)
        merged_data = merge_weather_and_power_data(weather_processed, power_processed)
        feature_engineered_data = create_lagged_features(merged_data)
        
        # Step 3: Exploratory Data Analysis
        print("\n--- Step 3: Exploratory Data Analysis ---\n")
        
        try:
            plot_target_distribution(feature_engineered_data, 'solar_generation')
            plot_target_distribution(feature_engineered_data, 'wind_generation')
            plot_correlation_heatmap(feature_engineered_data, 'solar_generation')
            plot_correlation_heatmap(feature_engineered_data, 'wind_generation')
        except Exception as e:
            print(f"Error in EDA: {e}")

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
        for model_type in ['linear_regression', 'random_forest', 'gradient_boosting', 'arima', 'prophet']:
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
        for model_type in ['linear_regression', 'random_forest', 'gradient_boosting', 'arima', 'prophet']:
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
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()