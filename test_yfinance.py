import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_spy_data():
    """Test fetching SPY data from Yahoo Finance"""
    print("\n=== Testing SPY Data Fetch ===\n")
    
    try:
        print("Fetching SPY data...")
        spy = yf.Ticker("SPY")
        current_data = spy.history(period="1d", interval="1m")
        
        if current_data.empty:
            print("Error: No data received from Yahoo Finance")
            return False
        
        current_price = current_data['Close'].iloc[-1]
        prev_price = current_data['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        print("\nData Summary:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Price Change: {price_change:+.2f}%")
        print(f"Data Points: {len(current_data)}")
        print(f"Time Range: {current_data.index[0]} to {current_data.index[-1]}")
        
        print("\nLatest Data Points:")
        print(current_data.tail())
        
        return True
        
    except Exception as e:
        print(f"\nError fetching SPY data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_spy_data()
