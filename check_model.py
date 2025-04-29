import joblib
import pandas as pd
from model_trainer import create_features

def check_model():
    """Check model and feature details"""
    print("\n=== Checking Model Configuration ===\n")
    
    try:
        # Load model and dependencies
        print("Loading model and dependencies...")
        model = joblib.load('spy_predictor.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        
        print("\nFeature Columns:")
        for i, col in enumerate(feature_columns, 1):
            print(f"{i}. {col}")
        
        # Load sample data
        print("\nLoading sample data...")
        df = pd.read_csv('HistoricalData_1745877114806.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.rename(columns={'Close/Last': 'Close'})
        
        # Create features for sample
        print("\nCreating sample features...")
        features_df = create_features(df.tail(10))
        
        print("\nAvailable columns in features:")
        for col in features_df.columns:
            print(f"- {col}")
            
        print("\nModel Details:")
        print(f"Number of features expected: {len(feature_columns)}")
        if hasattr(model, 'feature_importances_'):
            importances = sorted(zip(feature_columns, model.feature_importances_), 
                              key=lambda x: x[1], reverse=True)
            print("\nTop 10 important features:")
            for feat, imp in importances[:10]:
                print(f"{feat}: {imp:.4f}")
        
    except Exception as e:
        print(f"\nError checking model: {str(e)}")

if __name__ == "__main__":
    check_model()
