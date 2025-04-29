import pandas as pd
import numpy as np
from datetime import datetime

def validate_historical_data(file_path):
    """Validate and clean historical data"""
    print(f"Validating data from {file_path}")
    
    # Read the data
    df = pd.read_csv(file_path)
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Map column names
    column_mapping = {
        'Close/Last': 'Close',
        'Volume': 'Volume',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Verify required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows without intraday information
    has_intraday = (df['High'] != df['Low'])
    df_clean = df[has_intraday].copy()
    
    # Sort by date
    df_clean = df_clean.sort_values('Date')
    
    # Save cleaned data
    cleaned_file = file_path.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(cleaned_file, index=False)
    
    print(f"\nData Validation Results:")
    print(f"Original data rows: {len(df)}")
    print(f"Cleaned data rows: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} rows without intraday information")
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    
    # Print sample statistics
    print("\nSample Statistics:")
    print(f"Average daily range: ${(df_clean['High'] - df_clean['Low']).mean():.2f}")
    print(f"Average volume: {df_clean['Volume'].mean():.0f}")
    
    return df_clean

if __name__ == "__main__":
    validate_historical_data('HistoricalData_1745877114806.csv')
