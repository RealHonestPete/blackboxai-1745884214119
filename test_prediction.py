import pandas as pd
import numpy as np
import joblib
from model_trainer import create_features
from datetime import datetime, timedelta

def map_features(features_df):
    """Map feature names to match model expectations"""
    mapping = {
        'Price_Range': 'DailyRange',
        'Body_Size': 'BodySize',
        'Upper_Shadow': 'UpperShadow',
        'Lower_Shadow': 'LowerShadow',
        'Volatility_5min': 'Volatility_1d',
        'Volatility_15min': 'Volatility_5d'
    }
    
    df = features_df.copy()
    for new_name, old_name in mapping.items():
        if new_name in df.columns:
            df[old_name] = df[new_name]
    return df

def prepare_features(features_df, feature_columns):
    """Prepare features by handling missing values"""
    df = features_df.copy()
    
    # Forward fill NaN values
    df = df.ffill()
    
    # Back fill any remaining NaNs
    df = df.bfill()
    
    # If any NaNs still remain, fill with 0
    df = df.fillna(0)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            print(f"Missing column {col}, filling with zeros")
            df[col] = 0
            
    return df[feature_columns]

def analyze_scaled_features(features, scaler, feature_columns):
    """Analyze the scaled feature values"""
    # Get the raw values for the last row
    raw_values = features.iloc[-1]
    
    # Scale all features at once
    scaled_values = scaler.transform(features.iloc[-1:].values)[0]
    
    # Create a DataFrame with both raw and scaled values
    analysis = pd.DataFrame({
        'Feature': feature_columns,
        'Raw_Value': raw_values.values,
        'Scaled_Value': scaled_values
    })
    
    return analysis

def test_prediction():
    """Test the prediction functionality"""
    print("\n=== Testing Prediction System ===\n")
    
    try:
        # Load model and dependencies
        print("Loading model and dependencies...")
        model = joblib.load('spy_predictor.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        
        # Load sample data
        print("Loading test data...")
        df = pd.read_csv('HistoricalData_1745877114806.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.rename(columns={'Close/Last': 'Close'})
        
        # Take last 100 rows for testing
        test_data = df.tail(100).copy()
        
        # Create and map features
        print("Creating and mapping features...")
        features_df = create_features(test_data)
        features_df = map_features(features_df)
        
        # Prepare features
        print("Preparing features...")
        features = prepare_features(features_df, feature_columns)
        
        # Verify feature ranges
        print("\nFeature value verification:")
        for col in feature_columns:
            min_val = features[col].min()
            max_val = features[col].max()
            print(f"{col}: {min_val:.4f} to {max_val:.4f}")
        
        # Make predictions
        print("\nMaking predictions...")
        X = features.iloc[-1:].values
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        current_price = features_df['Close'].iloc[-1]
        
        # Calculate metrics
        print("\nPrediction Results:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        change_pct = ((prediction - current_price) / current_price * 100)
        print(f"Predicted Change: {change_pct:.2f}%")
        
        # Test prediction bounds
        reasonable_change = abs(change_pct) <= 5  # Within 5%
        print(f"\nPrediction Validation:")
        print(f"✓ Prediction generated successfully")
        print(f"{'✓' if reasonable_change else '✗'} Prediction within reasonable bounds")
        
        # Analyze scaled features
        print("\nFeature Analysis:")
        analysis = analyze_scaled_features(features, scaler, feature_columns)
        
        # Sort by feature importance
        importances = pd.Series(model.feature_importances_, index=feature_columns)
        analysis['Importance'] = importances
        analysis = analysis.sort_values('Importance', ascending=False)
        
        # Print top features
        print("\nTop 5 Most Influential Features:")
        for _, row in analysis.head().iterrows():
            print(f"\n{row['Feature']}:")
            print(f"  Raw value: {row['Raw_Value']:.4f}")
            print(f"  Scaled value: {row['Scaled_Value']:.4f}")
            print(f"  Importance: {row['Importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\nError in prediction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_prediction()
