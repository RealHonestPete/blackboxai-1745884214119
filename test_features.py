import pandas as pd
import numpy as np
import joblib
from model_trainer import create_features

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

def test_features():
    """Test feature generation and mapping"""
    print("\n=== Testing Feature Generation ===\n")
    
    try:
        # Load model requirements
        print("Loading model dependencies...")
        feature_columns = joblib.load('feature_columns.joblib')
        print(f"Model expects {len(feature_columns)} features")
        
        # Load sample data
        print("\nLoading sample data...")
        df = pd.read_csv('HistoricalData_1745877114806.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.rename(columns={'Close/Last': 'Close'})
        
        # Generate features
        print("Generating features...")
        features_df = create_features(df.tail(100))
        
        # Map features
        print("Mapping feature names...")
        features_df = map_features(features_df)
        
        # Check for missing features
        missing_features = [col for col in feature_columns if col not in features_df.columns]
        if missing_features:
            print("\nMissing features:")
            for feat in missing_features:
                print(f"- {feat}")
        else:
            print("\nAll required features are present!")
        
        # Print available features
        print("\nAvailable features:")
        for col in sorted(features_df.columns):
            if col in feature_columns:
                print(f"✓ {col}")
            else:
                print(f"  {col}")
        
        # Sample values
        print("\nSample feature values (last row):")
        for col in feature_columns:
            if col in features_df.columns:
                val = features_df[col].iloc[-1]
                print(f"{col}: {val:.4f}")
        
        return features_df, feature_columns
        
    except Exception as e:
        print(f"\nError in feature test: {str(e)}")
        return None, None

if __name__ == "__main__":
    features_df, feature_columns = test_features()
