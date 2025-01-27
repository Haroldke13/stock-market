import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, flash, redirect
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
"""from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping"""
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'joelcodesecretkey'

# Enable logging for debugging
#logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    """Home route to render the landing page."""
    return render_template('home.html')


@app.route('/yfinance', methods=['GET', 'POST'])
def yfinance_route():
    """Fetch stock data and display it in a table."""
    if request.method == 'POST':
        stock_symbol = request.form.get('stock')
        
        if not stock_symbol:
            return render_template('yfinance.html', error="Stock symbol is required.",stock_symbol=stock_symbol)
        
        try:
            # Fetch stock data from yfinance
            stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
            
            if stock_data.empty:
                return render_template('yfinance.html', error=f"No data found for symbol '{stock_symbol}'.")
            
            # Reset index to make Date a column
            stock_data.reset_index(inplace=True)

            # Pass data to template
            stock_data_dict = stock_data.to_dict(orient='records')
            columns = stock_data.columns.tolist()

            return render_template(
                'yfinance.html',
                stock_symbol=stock_symbol,
                stock_data=stock_data_dict,
                columns=columns
            )
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return render_template('yfinance.html', error=f"An error occurred: {str(e)}",stock_symbol=stock_symbol)

    return render_template('yfinance.html')

"""
@app.route('/stock_market', methods=['POST', 'GET'])
def stock_market():
    #Perform stock market prediction using LSTM.
    stock_symbol = request.form.get('stock_symbol')

    if not stock_symbol:
        flash("Stock symbol is required.", "danger")
        return redirect('/yfinance')

    try:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        if stock_data.empty:
            flash("No data available for this stock symbol.", "danger")
            return redirect('/yfinance')

        stock_data.reset_index(inplace=True)

        # Clean and preprocess data
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        data = stock_data[['Close']]

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare data for LSTM
        def create_dataset(data, time_step=60):
            x, y = [], []
            for i in range(time_step, len(data)):
                x.append(data[i - time_step:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len]
        test_data = scaled_data[training_data_len - 60:]

        x_train, y_train = create_dataset(train_data)
        x_test, y_test = create_dataset(test_data)

        # Reshape data for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Future price prediction
        last_60_days = scaled_data[-60:]
        future_input = np.reshape(last_60_days, (1, 60, 1))
        future_price = scaler.inverse_transform(model.predict(future_input))[0][0]

        # Prepare performance report
        report = {
            'symbol': stock_symbol,
            'latest_price': stock_data['Close'].iloc[-1],
            'predicted_next_price': future_price,
            'accuracy': 100 - np.mean(np.abs(predictions - scaler.inverse_transform(test_data[60:])) / scaler.inverse_transform(test_data[60:])) * 100
        }

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index[-len(test_data):], scaler.inverse_transform(test_data[60:]), label='Actual Prices', color='blue')
        ax.plot(stock_data.index[-len(predictions):], predictions, label='Predicted Prices', linestyle='--', color='red')
        ax.axvline(stock_data.index[-1], color='green', linestyle=':', label='Prediction Point')
        ax.set_title(f'{stock_symbol} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig)

        return render_template('stock_market.html', report=report, stock_symbol=stock_symbol, img_data=img_data)

    except Exception as e:
        logging.error(f"Error processing stock market data: {e}")
        flash(f"Error fetching or processing data: {str(e)}", "danger")
        return redirect('/yfinance')
"""

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

@app.route('/stock_market', methods=['POST', 'GET'])
def stock_market():
    """Perform stock market prediction using ARIMA."""
    stock_symbol = request.form.get('stock_symbol')

    if not stock_symbol:
        flash("Stock symbol is required.", "danger")
        return redirect('/yfinance')

    try:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        if stock_data.empty:
            flash("No data available for this stock symbol.", "danger")
            return redirect('/yfinance')

        stock_data.reset_index(inplace=True)

        # Clean and preprocess data
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        data = stock_data['Close']

        # Split data into training and testing sets (80-20 split)
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Fit ARIMA model
        model = ARIMA(train_data, order=(5, 1, 0))  # (p, d, q) parameters can be tuned
        arima_model = model.fit()

        # Make predictions
        predictions = arima_model.forecast(steps=len(test_data))
        accuracy = 100 - mean_absolute_percentage_error(test_data, predictions) * 100

        # Predict the next day's price
        future_prediction = arima_model.forecast(steps=1)[0]

        # Prepare performance report
        report = {
            'symbol': stock_symbol,
            'latest_price': data.iloc[-1],
            'predicted_next_price': future_prediction,
            'accuracy': accuracy
        }

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data, label='Actual Prices', color='blue')
        ax.plot(test_data.index, predictions, label='Predicted Prices', linestyle='--', color='red')
        ax.axvline(test_data.index[0], color='green', linestyle=':', label='Prediction Start Point')
        ax.set_title(f'{stock_symbol} Price Prediction (ARIMA)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig)

        return render_template('stock_market.html', report=report, stock_symbol=stock_symbol, img_data=img_data)

    except Exception as e:
        logging.error(f"Error processing stock market data: {e}")
        flash(f"Error fetching or processing data: {str(e)}", "danger")
        return redirect('/yfinance')





if __name__ == '__main__':
    app.run()
