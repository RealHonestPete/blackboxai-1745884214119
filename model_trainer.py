import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import pandas_datareader as pdr

def fetch_bond_data():
    """Fetch US Treasury bond yields"""
    try:
        bonds = pdr.get_data_fred([
            'DGS2', 'DGS5', 'DGS10', 'DGS30'
        ], start=datetime.now() - timedelta(days=30))
        bonds.index = bonds.index.tz_localize('America/New_York').tz_convert('UTC')
        bonds = bonds.ffill()
        bonds['2Y5Y_Spread'] = bonds['DGS5'] - bonds['DGS2']
        bonds['5Y10Y_Spread'] = bonds['DGS10'] - bonds['DGS5']
        bonds['10Y30Y_Spread'] = bonds['DGS30'] - bonds['DGS10']
        bonds['2Y10Y_Spread'] = bonds['DGS10'] - bonds['DGS2']
        return bonds
    except Exception as e:
        print(f"Error fetching bond data: {str(e)}")
        return None

def fetch_historical_data():
    """Fetch SPY data with 1-minute intervals for multiple chunks"""
    spy = yf.Ticker("SPY")
    end_date = datetime.now()
    
    # Get 1-minute data for the last 7 days
    df_minute = spy.history(start=end_date - timedelta(days=7), end=end_date, interval='1m')
    if not df_minute.empty:
        df_minute.index = pd.to_datetime(df_minute.index).tz_convert('UTC')
    
    # Get hourly data for additional historical context
    df_hourly = spy.history(start=end_date - timedelta(days=30), end=end_date - timedelta(days=7), interval='1h')
    if not df_hourly.empty:
        df_hourly.index = pd.to_datetime(df_hourly.index).tz_convert('UTC')
        # Resample hourly data to 1-minute to match the recent data
        df_hourly = df_hourly.resample('1min').ffill()
    
    # Combine the datasets
    df = pd.concat([df_hourly, df_minute])
    
    # Filter for market hours (9:30 AM to 4:00 PM ET)
    df.index = df.index.tz_convert('America/New_York')
    market_hours = (df.index.time >= pd.Timestamp('09:30').time()) & \
                  (df.index.time <= pd.Timestamp('16:00').time()) & \
                  (df.index.dayofweek < 5)  # Monday = 0, Friday = 4
    df = df[market_hours]
    df.index = df.index.tz_convert('UTC')
    
    return df

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(period - 1) + upval)/period
        down = (down*(period - 1) + downval)/period
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, window=20):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, ma, lower

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def create_features(df, bond_data=None):
    # Copy the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['PrevClose'] = df['Close'].shift(1)
    
    # Moving averages and ratios
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA{window}_Ratio'] = df['Close'] / df[f'MA{window}']
        df[f'MA{window}_Slope'] = df[f'MA{window}'].diff()
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close'].values)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Volatility indicators
    df['Volatility_1min'] = df['Returns'].rolling(window=10).std()
    df['Volatility_5min'] = df['Returns'].rolling(window=50).std()
    df['Volatility_15min'] = df['Returns'].rolling(window=150).std()
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['Bollinger_Middle']
    
    # Volume indicators
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA10']
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    # Price patterns
    df['Price_Range'] = df['High'] - df['Low']
    df['Body_Size'] = abs(df['Open'] - df['Close'])
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Momentum indicators
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic_oscillator(df)
    df['ATR'] = calculate_atr(df)
    
    # Price levels
    df['Distance_From_High'] = df['High'].rolling(window=20).max() - df['Close']
    df['Distance_From_Low'] = df['Close'] - df['Low'].rolling(window=20).min()
    
    if bond_data is not None:
        bond_data_resampled = bond_data.reindex(df.index, method='ffill')
        df['Treasury_2Y'] = bond_data_resampled['DGS2']
        df['Treasury_5Y'] = bond_data_resampled['DGS5']
        df['Treasury_10Y'] = bond_data_resampled['DGS10']
        df['Treasury_30Y'] = bond_data_resampled['DGS30']
        df['Spread_2Y5Y'] = bond_data_resampled['2Y5Y_Spread']
        df['Spread_5Y10Y'] = bond_data_resampled['5Y10Y_Spread']
        df['Spread_10Y30Y'] = bond_data_resampled['10Y30Y_Spread']
        df['Spread_2Y10Y'] = bond_data_resampled['2Y10Y_Spread']
        for col in ['DGS2', 'DGS5', 'DGS10', 'DGS30']:
            df[f'{col}_Change'] = bond_data_resampled[col].pct_change()
    
    # Time-based features
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['TimeFromOpen'] = (df.index.hour * 60 + df.index.minute - 570)
    df['IsRegularHours'] = ((df.index.hour * 60 + df.index.minute >= 570) & 
                           (df.index.hour * 60 + df.index.minute < 960)).astype(int)
    
    df['Target'] = df['Close'].shift(-10)
    
    return df

def prepare_data(df):
    feature_columns = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    df = df.dropna()
    X = df[feature_columns]
    y = df['Target']
    # Convert all column names to strings to avoid mixed types
    X.columns = X.columns.astype(str)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    return X_scaled, y, feature_columns

def evaluate_predictions(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': round(rmse, 3), 'MAE': round(mae, 3), 'MAPE': round(mape, 3)}

def train_model():
    print("Fetching historical data...")
    df = fetch_historical_data()
    if df.empty:
        raise Exception("No data received from Yahoo Finance")
    print(f"Received {len(df)} SPY data points")
    
    print("Fetching bond market data...")
    bond_data = fetch_bond_data()
    if bond_data is not None:
        print(f"Received bond market data with {len(bond_data)} points")
    
    print("Creating features...")
    df = create_features(df, bond_data)
    
    print("Preparing data...")
    X, y, feature_columns = prepare_data(df)
    print(f"Using features: {feature_columns}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Training model...")
    # Improved model parameters
    model = GradientBoostingRegressor(
        n_estimators=500,  # Increased from 200
        learning_rate=0.05,  # Decreased from 0.1 for better generalization
        max_depth=6,  # Increased from 5
        subsample=0.8,
        min_samples_split=100,  # Added to prevent overfitting
        min_samples_leaf=20,  # Added to prevent overfitting
        random_state=42
    )
    
    cv_metrics = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        cv_metrics.append(metrics)
    
    avg_metrics = {metric: np.mean([fold[metric] for fold in cv_metrics]) for metric in cv_metrics[0].keys()}
    print("\nAverage Cross-validation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Train final model on all data
    model.fit(X, y)
    
    # Print feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    print("\nTop 10 Important Features:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feature}: {importance:.3f}")
    
    joblib.dump(model, 'spy_predictor.joblib')
    return model

if __name__ == "__main__":
    train_model()
