import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from datetime import datetime, timedelta

def prepare_data(df):
    """Prepare data for training"""
    print("\nPreparing data...")
    
    # Rename Close/Last to Close
    df = df.rename(columns={'Close/Last': 'Close'})
    
    # Calculate basic features
    df['Returns'] = df['Close'].pct_change()
    df['PrevClose'] = df['Close'].shift(1)
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA{window}_Ratio'] = df['Close'] / df[f'MA{window}']
    
    # Volatility
    df['Volatility_1d'] = df['Returns'].rolling(window=10).std()
    df['Volatility_5d'] = df['Returns'].rolling(window=50).std()
    
    # Price patterns
    df['DailyRange'] = df['High'] - df['Low']
    df['BodySize'] = abs(df['Open'] - df['Close'])
    df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Volume indicators
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target (10-minute future return)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train the prediction model"""
    print("Training model...")
    
    # Define features
    feature_columns = [
        'Returns', 'PrevClose',
        'MA5', 'MA5_Ratio',
        'MA10', 'MA10_Ratio',
        'MA20', 'MA20_Ratio',
        'MA50', 'MA50_Ratio',
        'Volatility_1d', 'Volatility_5d',
        'DailyRange', 'BodySize',
        'UpperShadow', 'LowerShadow',
        'Volume_MA10', 'Volume_Ratio',
        'RSI'
    ]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    print("\nSaving model artifacts...")
    joblib.dump(model, 'spy_predictor.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    
    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False)
    
    print("\nTop 10 Important Features:")
    for feat, imp in importances[:10].items():
        print(f"{feat}: {imp:.4f}")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    print("=== Retraining Model ===")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('HistoricalData_1745877114806.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Prepare and train
    df = prepare_data(df)
    model, scaler, features = train_model(df)
    
    print("\nModel retraining complete!")
