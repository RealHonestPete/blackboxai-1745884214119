
Built by https://www.blackbox.ai

---

```markdown
# SPY Price Predictor

## Project Overview
SPY Price Predictor is an application designed to predict the future price of the SPDR S&P 500 ETF (SPY). Utilizing machine learning techniques and real-time data fetching, it provides users with predictions of the SPY price in the next 10 minutes, along with historical and accuracy metrics. The application features an intuitive interface built with HTML, CSS (using Tailwind), and JavaScript to provide a seamless user experience.

## Installation
To run the SPY Price Predictor locally, please follow these steps: 

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/spy-price-predictor.git
   ```

2. **Navigate into the project directory**:
   ```bash
   cd spy-price-predictor
   ```

3. **Set up a Python environment** and install the necessary packages:
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. **Install additional required packages**:
   If additional packages are needed for your environment or if you want to run the training script and model simulations, install those as necessary (e.g. `yfinance`, `numpy`, `pandas`, `scikit-learn`, `joblib`).

5. **Run the server** to host the API:
   ```bash
   python server.py
   ```

6. **Open the application** by navigating to `index.html` in your browser.

## Usage
- The application displays a prediction of the SPY price for the next 10 minutes, as well as the current price and percentage change.
- Users can view accuracy metrics related to the model's predictions.
- The application fetches real-time data from a local Python server that utilizes Yahoo Finance data to get the latest SPY prices and predictions.

## Features
- **10-Minute Price Predictions**: Get up-to-date predictions on SPY price movements.
- **Current Price Display**: Shows the current SPY price and its recent change in percentage.
- **Performance Metrics**: Displays the latest accuracy metrics based on the model predictions.
- **Responsive Design**: Built with Tailwind CSS, making the interface clean and adaptive to various screen sizes.

## Dependencies
The project uses the following dependencies as specified in `package.json`:
- For frontend styles: [Tailwind CSS](https://tailwindcss.com/)
- Included FontAwesome for icons.
- JavaScript libraries for handling data fetching and DOM manipulation.

Python backend dependencies:
- `yfinance` for fetching stock data.
- `pandas`, `numpy` for data manipulation.
- `scikit-learn` for modeling and predictions.
- `joblib` for saving and loading models.

## Project Structure
```plaintext
spy-price-predictor/
├── index.html                # Main entry point of the application.
├── styles.css                # Custom styles for the application.
├── script.js                 # JavaScript file for handling data fetching and UI updates.
├── server.py                 # Python server to fetch and serve SPY data.
├── model_trainer.py          # Script for training the price prediction model.
├── simulation.py             # Simulation script to test prediction accuracy.
├── historical_trainer.py     # Script to handle historical data training for the model.
├── godel_client.py           # WebSocket client for real-time price updates (if used).
├── simulation_with_kaggle_data.py # Simulation on Kaggle data for additional testing.
└── requirements.txt          # Python dependencies for the project.
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
```