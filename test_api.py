import requests
import json
import time

def test_api():
    """Test the SPY data API endpoint"""
    print("\n=== Testing API Endpoint ===\n")
    
    url = "http://localhost:8000/api/spy-data"
    
    try:
        print(f"Making request to {url}...")
        response = requests.get(url)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\nResponse Data:")
            print(json.dumps(data, indent=2))
            
            # Verify required fields
            required_fields = [
                'current_price',
                'price_change',
                'prediction',
                'accuracy_metrics',
                'lastUpdate'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"\nWarning: Missing fields: {missing_fields}")
            else:
                print("\nAll required fields present!")
                
            # Print key metrics
            print("\nKey Metrics:")
            print(f"Current Price: ${data['current_price']}")
            print(f"Price Change: {data['price_change']:+.2f}%")
            print(f"Predicted Price: ${data['prediction']['target_price']}")
            print(f"Last Update: {data['lastUpdate']}")
            
            if data.get('accuracy_metrics'):
                print("\nAccuracy Metrics:")
                metrics = data['accuracy_metrics']
                print(f"Hour Accuracy: {metrics['hour_accuracy']}%")
                print(f"Today Accuracy: {metrics['today_accuracy']}%")
                print(f"Average Error: ${metrics['avg_error']}")
                
            return True
            
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\nError testing API: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Wait a bit for server to be ready
    time.sleep(2)
    test_api()
