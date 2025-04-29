import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import subprocess
import time

class TestSuite:
    def __init__(self):
        self.test_results = []
        
    def run_all_tests(self):
        """Run all test cases"""
        print("\n=== Starting Test Suite ===\n")
        
        # 1. Data Validation Tests
        self.test_data_integrity()
        
        # 2. Model Training Tests
        self.test_model_training()
        
        # 3. Server Tests
        self.test_server()
        
        # 4. Frontend Tests
        self.test_frontend()
        
        # Print Results
        self.print_results()
    
    def test_data_integrity(self):
        """Test data validation and cleaning"""
        try:
            print("Testing data integrity...")
            
            # Load data
            df = pd.read_csv('HistoricalData_1745877114806.csv')
            
            # Test cases
            tests = [
                ('Data not empty', len(df) > 0),
                ('Required columns present', all(col in df.columns for col in ['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low'])),
                ('No missing values', df.isnull().sum().sum() == 0),
                ('Date format valid', pd.to_datetime(df['Date']).notnull().all()),
                ('Price values numeric', all(df[col].dtype in ['float64', 'int64'] for col in ['Close/Last', 'Open', 'High', 'Low'])),
                ('Volume values numeric', df['Volume'].dtype in ['float64', 'int64']),
                ('High >= Low', (df['High'] >= df['Low']).all())
            ]
            
            for test_name, result in tests:
                self.test_results.append(('Data Integrity', test_name, result))
                
        except Exception as e:
            print(f"Error in data integrity tests: {str(e)}")
            self.test_results.append(('Data Integrity', 'Exception occurred', False))
    
    def test_model_training(self):
        """Test model training functionality"""
        try:
            print("\nTesting model training...")
            
            # Test model trainer imports
            import model_trainer
            
            # Test cases
            tests = [
                ('Model trainer module exists', True),
                ('Required functions present', all(hasattr(model_trainer, func) for func in ['create_features', 'train_model'])),
                ('Model file exists', os.path.exists('spy_predictor.joblib')),
                ('Scaler file exists', os.path.exists('scaler.joblib')),
                ('Feature columns file exists', os.path.exists('feature_columns.joblib'))
            ]
            
            for test_name, result in tests:
                self.test_results.append(('Model Training', test_name, result))
                
        except Exception as e:
            print(f"Error in model training tests: {str(e)}")
            self.test_results.append(('Model Training', 'Exception occurred', False))
    
    def test_server(self):
        """Test server functionality"""
        try:
            print("\nTesting server...")
            
            # Start server in background
            server_process = subprocess.Popen(['python3', 'server.py'])
            time.sleep(2)  # Give server time to start
            
            # Test cases
            tests = [
                ('Server starts successfully', server_process.poll() is None),
                ('Required files present', all(os.path.exists(f) for f in ['server.py', 'model_trainer.py'])),
            ]
            
            for test_name, result in tests:
                self.test_results.append(('Server', test_name, result))
            
            # Clean up
            server_process.terminate()
            
        except Exception as e:
            print(f"Error in server tests: {str(e)}")
            self.test_results.append(('Server', 'Exception occurred', False))
    
    def test_frontend(self):
        """Test frontend files and structure"""
        try:
            print("\nTesting frontend...")
            
            # Test cases
            tests = [
                ('HTML file exists', os.path.exists('index.html')),
                ('CSS file exists', os.path.exists('styles.css')),
                ('JavaScript file exists', os.path.exists('script.js')),
                ('HTML contains required elements', self.check_html_content()),
                ('CSS contains required styles', self.check_css_content()),
                ('JavaScript contains required functions', self.check_js_content())
            ]
            
            for test_name, result in tests:
                self.test_results.append(('Frontend', test_name, result))
                
        except Exception as e:
            print(f"Error in frontend tests: {str(e)}")
            self.test_results.append(('Frontend', 'Exception occurred', False))
    
    def check_html_content(self):
        """Verify HTML content"""
        try:
            with open('index.html', 'r') as f:
                content = f.read()
                return all(x in content for x in ['<!DOCTYPE html>', '<script src="script.js">'])
        except:
            return False
    
    def check_css_content(self):
        """Verify CSS content"""
        try:
            with open('styles.css', 'r') as f:
                content = f.read()
                return all(x in content for x in ['.card', '@media'])
        except:
            return False
    
    def check_js_content(self):
        """Verify JavaScript content"""
        try:
            with open('script.js', 'r') as f:
                content = f.read()
                return all(x in content for x in ['fetchSPYData', 'updateUI'])
        except:
            return False
    
    def print_results(self):
        """Print test results summary"""
        print("\n=== Test Results ===\n")
        
        current_component = None
        passed = 0
        total = 0
        
        for component, test_name, result in self.test_results:
            if component != current_component:
                if current_component is not None:
                    print()
                print(f"{component} Tests:")
                current_component = component
            
            status = "✓" if result else "✗"
            print(f"{status} {test_name}")
            
            if result:
                passed += 1
            total += 1
        
        print(f"\nOverall: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")

if __name__ == "__main__":
    test_suite = TestSuite()
    test_suite.run_all_tests()
