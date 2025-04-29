import pandas as pd
import os
import subprocess
import time

def test_data():
    """Test data validation"""
    print("\nTesting data...")
    try:
        # Load data
        df = pd.read_csv('HistoricalData_1745877114806.csv')
        
        # Run tests
        tests = {
            'Data loaded successfully': len(df) > 0,
            'Required columns present': all(col in df.columns for col in ['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low']),
            'No missing values': df.isnull().sum().sum() == 0,
            'High >= Low': (df['High'] >= df['Low']).all()
        }
        
        # Print results
        print("\nData Test Results:")
        passed = 0
        for test_name, result in tests.items():
            status = "✓" if result else "✗"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        print(f"\nData Tests: {passed}/{len(tests)} passed")
        
    except Exception as e:
        print(f"Error in data tests: {str(e)}")

def test_files():
    """Test required files existence"""
    print("\nTesting required files...")
    
    required_files = {
        'Frontend': ['index.html', 'styles.css', 'script.js'],
        'Backend': ['server.py', 'model_trainer.py'],
        'Model': ['spy_predictor.joblib', 'scaler.joblib', 'feature_columns.joblib']
    }
    
    total_passed = 0
    total_tests = 0
    
    for category, files in required_files.items():
        print(f"\n{category} Files:")
        passed = 0
        for file in files:
            exists = os.path.exists(file)
            status = "✓" if exists else "✗"
            print(f"{status} {file}")
            if exists:
                passed += 1
                total_passed += 1
            total_tests += 1
        print(f"{passed}/{len(files)} {category} files present")
    
    print(f"\nFile Tests: {total_passed}/{total_tests} passed")

def kill_port_8000():
    """Kill any process using port 8000"""
    try:
        subprocess.run(['pkill', '-f', 'python.*8000'], shell=True)
        time.sleep(1)
    except:
        pass

if __name__ == "__main__":
    print("=== Running Core Tests ===")
    
    # Kill any process on port 8000
    kill_port_8000()
    
    # Run tests
    test_data()
    test_files()
