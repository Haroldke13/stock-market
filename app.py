import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, flash, redirect,send_from_directory
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
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

import os

# Route to fetch stock data and save as CSV
@app.route('/yfinance', methods=['GET', 'POST'])
def yfinance_route():
    """Fetch stock data, save as CSV, and display it in a table."""
    if request.method == 'POST':
        stock_symbol = request.form.get('stock')
        
        if not stock_symbol:
            return render_template('yfinance.html', error="Stock symbol is required.", stock_symbol=stock_symbol)
        
        try:
            # Fetch stock data from yfinance
            stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
            
            if stock_data.empty:
                return render_template('yfinance.html', error=f"No data found for symbol '{stock_symbol}'.")
            
            # Reset index to make Date a column
            stock_data.reset_index(inplace=True)

            # Save stock data to a CSV file in the root directory or static directory
            csv_filename = f"{stock_symbol}_stock_data.csv"
            csv_filepath = os.path.join(os.getcwd(), csv_filename)
            stock_data.to_csv(csv_filepath, index=False)
            print(f"File saved at: {csv_filepath}")  # Debugging

            # Pass data to template
            stock_data_dict = stock_data.to_dict(orient='records')
            columns = stock_data.columns.tolist()

            return render_template(
                'yfinance.html',
                stock_symbol=stock_symbol,
                stock_data=stock_data_dict,
                columns=columns,
                csv_filename=csv_filename,
                message=f"Data saved as {csv_filename} at {csv_filepath}. You can access it in the stock market prediction tab above for prediction"
            )
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return render_template('yfinance.html', error=f"An error occurred: {str(e)}", stock_symbol=stock_symbol)

    return render_template('yfinance.html')


# Route to download CSV file
@app.route('/download_csv1/<filename>')
def download_csv1(filename):
    """Allow the user to download the CSV file."""
    return send_from_directory(os.getcwd(), filename, as_attachment=True)


import os
import pandas as pd
import yfinance as yf
from flask import render_template, request
import plotly.graph_objs as go
import json
import logging


@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    """Analyze and clean the CSV data, perform prediction, and store it in the database."""
    csv_filename = request.form.get('analyze')
    
    if not csv_filename:
        return render_template('yfinance.html', error="No CSV file selected for analysis.")
    
    try:
        # Read the CSV file
        csv_filepath = os.path.join(os.getcwd(), csv_filename)
        stock_data = pd.read_csv(csv_filepath)
        
        # Validate columns in the uploaded CSV (like Date, Close)
        if 'Date' not in stock_data.columns or 'Close' not in stock_data.columns:
            return render_template('yfinance.html', error="CSV must contain 'Date' and 'Close' columns.")
        
        # Data cleaning
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Date', 'Close'])  # Drop rows with invalid dates or missing data
        stock_data.set_index('Date', inplace=True)

        # Ensure the 'Close' column is numeric
        stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Close'])  # Drop rows with invalid or NaN 'Close' values

        # Check if we have enough data for prediction
        if len(stock_data) < 50:
            return render_template('yfinance.html', error="Not enough data for reliable analysis.")

        # Perform statistical analysis or prediction (e.g., ARIMA)
        closing_prices = stock_data['Close']
        train_data = closing_prices[:-30]
        test_data = closing_prices[-30:]

        # Simple ARIMA model prediction example
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train_data, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecasting for the next 30 days
        predictions = model_fit.forecast(steps=30)

        # Create a DataFrame for the prediction results
        results = pd.DataFrame({
            'Date': closing_prices.index[-30:],
            'Actual': test_data.values,
            'Predicted': predictions
        })

        # Calculate Exponential Moving Averages (EMAs) for analysis
        ema_actual_21 = results['Actual'].ewm(span=21, adjust=False).mean()
        ema_predicted_21 = results['Predicted'].ewm(span=21, adjust=False).mean()

        ema_actual_50 = results['Actual'].ewm(span=50, adjust=False).mean()
        ema_predicted_50 = results['Predicted'].ewm(span=50, adjust=False).mean()

        ema_actual_100 = results['Actual'].ewm(span=100, adjust=False).mean()
        ema_predicted_100 = results['Predicted'].ewm(span=100, adjust=False).mean()

        # Determine trend (Bullish or Bearish) based on EMA
        trend = "Bullish" if results['Predicted'].iloc[-1] > ema_predicted_21.iloc[-1] else "Bearish"

        # Convert the 'Date' column to strings for rendering in the template
        results['Date'] = results['Date'].dt.strftime('%Y-%m-%d')
        results_dict = results.to_dict(orient='records')

        # Create the candlestick chart with Plotly
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Candlestick'
        )])

        # Add the predicted values as a line graph
        fig.add_trace(go.Scatter(
            x=results['Date'], 
            y=results['Predicted'], 
            mode='lines', 
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        # Add the 21, 50, and 100-day EMAs to the chart
        fig.add_trace(go.Scatter(
            x=results['Date'], 
            y=ema_predicted_21, 
            mode='lines', 
            name='EMA 21 Predicted',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=results['Date'], 
            y=ema_predicted_50, 
            mode='lines', 
            name='EMA 50 Predicted',
            line=dict(color='green', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=results['Date'], 
            y=ema_predicted_100, 
            mode='lines', 
            name='EMA 100 Predicted',
            line=dict(color='red', width=2)
        ))

        # Customize layout of the plot
        fig.update_layout(
            title="Stock Prediction with ARIMA",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        # Convert the plot to JSON for rendering in the template
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'yfinance.html',
            stock_symbol=csv_filename.split('.')[0],  # Display the filename (without extension)
            results=results_dict,
            columns=list(results.columns),
            trend=trend,
            graph_json=graph_json,
            message="Analysis completed successfully."
        )
    
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return render_template('yfinance.html', error=f"An error occurred during analysis: {str(e)}")


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import os
from flask import send_file

UPLOAD_FOLDER = os.path.abspath(os.getcwd())
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
import plotly.graph_objects as go
import json
import plotly

@app.route('/stock_market_prediction', methods=['GET', 'POST'])
def stock_market_prediction():
    csv_files = [file for file in os.listdir(UPLOAD_FOLDER) if file.endswith('.csv')]  # Filter only CSV files

    if request.method == 'POST':
        stock_symbol = request.form.get('stock')
        uploaded_file = request.files.get('file')

        # Ensure at least one input is provided
        if not stock_symbol and not uploaded_file:
            return render_template(
                'stock_market_prediction.html',
                error="Please provide either a stock symbol or upload a CSV file.",
                csv_files=csv_files  # Pass filtered CSV files list
            )

        try:
            if uploaded_file:
                # Handle uploaded CSV file
                csv_filename = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(csv_filename)
                stock_data = pd.read_csv(csv_filename)

                # Validate columns in the uploaded CSV
                if 'Date' not in stock_data.columns or 'Close' not in stock_data.columns:
                    return render_template(
                        'stock_market_prediction.html',
                        error="Uploaded CSV must contain 'Date' and 'Close' columns.",
                        csv_files=csv_files  # Pass filtered CSV files list
                    )

                # Parse and index the date column
                stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
                stock_data = stock_data.dropna(subset=['Date', 'Close'])  # Drop rows with invalid dates or missing data
                stock_data.set_index('Date', inplace=True)

                # Clean string values from 'Close' column (convert to numeric)
                stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')

                # Drop rows with invalid or NaN 'Close' values
                stock_data = stock_data.dropna(subset=['Close'])
            else:
                # Handle stock symbol input
                stock_data = yf.download(
                    stock_symbol,
                    start='2010-01-01',
                    end=pd.Timestamp.today().strftime('%Y-%m-%d')
                )
                
                if stock_data.empty:
                    return render_template(
                        'stock_market_prediction.html',
                        error=f"No data found for symbol '{stock_symbol}'.",
                        csv_files=csv_files  # Pass filtered CSV files list
                    )

                # Save downloaded data as a CSV
                csv_filename = os.path.join(UPLOAD_FOLDER, f"{stock_symbol}_data.csv")
                stock_data.to_csv(csv_filename)

            # Ensure valid datetime index and frequency
            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data = stock_data.asfreq('B')  # Business days

            # Drop rows with missing or invalid data in the 'Close' column
            stock_data = stock_data.dropna(subset=['Close'])

            # Ensure sufficient data for ARIMA
            if len(stock_data['Close']) < 50:
                return render_template(
                    'stock_market_prediction.html',
                    error="Not enough data for reliable prediction.",
                    csv_files=csv_files  # Pass filtered CSV files list
                )

            # Train-test split for ARIMA
            closing_prices = stock_data['Close']
            train_data = closing_prices[:-30]
            test_data = closing_prices[-30:]

            # Train ARIMA model
            model = ARIMA(train_data, order=(5, 1, 0))
            model_fit = model.fit()

            # Forecasting
            predictions = model_fit.forecast(steps=30)

            # Prepare results for rendering
            results = pd.DataFrame({
                'Date': closing_prices.index[-30:],
                'Actual': test_data.values,
                'Predicted': predictions
            })

            # Calculate EMA for Actual and Predicted values
            ema_actual_21 = results['Actual'].ewm(span=21, adjust=False).mean()  # 21-day EMA
            ema_predicted_21 = results['Predicted'].ewm(span=21, adjust=False).mean()

            ema_actual_50 = results['Actual'].ewm(span=50, adjust=False).mean()  # 50-day EMA
            ema_predicted_50 = results['Predicted'].ewm(span=50, adjust=False).mean()

            ema_actual_100 = results['Actual'].ewm(span=100, adjust=False).mean()  # 100-day EMA
            ema_predicted_100 = results['Predicted'].ewm(span=100, adjust=False).mean()

            # Determine trend based on EMA
            if results['Predicted'].iloc[-1] > ema_predicted_21.iloc[-1]:
                trend = "Bullish"
            else:
                trend = "Bearish"

            # Convert the 'Date' column to strings for rendering
            results['Date'] = results['Date'].dt.strftime('%Y-%m-%d')
            results_dict = results.to_dict(orient='records')

            # Add trend to results
            results_dict.append({"Trend": trend})

            # Create the candlestick chart with Plotly
            fig = go.Figure(data=[go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            )])

            # Add the predicted values as a line graph
            fig.add_trace(go.Scatter(
                x=results['Date'], 
                y=results['Predicted'], 
                mode='lines', 
                name='Predicted',
                line=dict(color='orange', width=2)
            ))

            # Add the 21, 50, and 100-day EMAs to the chart
            fig.add_trace(go.Scatter(
                x=results['Date'], 
                y=ema_predicted_21, 
                mode='lines', 
                name='EMA 21 Predicted',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=results['Date'], 
                y=ema_predicted_50, 
                mode='lines', 
                name='EMA 50 Predicted',
                line=dict(color='green', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=results['Date'], 
                y=ema_predicted_100, 
                mode='lines', 
                name='EMA 100 Predicted',
                line=dict(color='red', width=2)
            ))

            # Customize layout
            fig.update_layout(
                title=f"Stock Prediction for {stock_symbol}",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                xaxis_rangeslider_visible=False
            )

            # Convert the plot to JSON
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Extract the filename (without extension) to display in the template
            filename_without_extension = uploaded_file.filename.rsplit('.', 1)[0]

            return render_template(
                'stock_market_prediction.html',
                stock_symbol=filename_without_extension,  # Pass the filename (without extension) here
                results=results_dict,
                columns=list(results.columns),
                csv_files=csv_files,  # Pass filtered CSV files list
                graph_json=graph_json,
                trend=trend
            )
        except Exception as e:
            logging.error(f"Error during stock market prediction: {e}")
            return render_template(
                'stock_market_prediction.html',
                error=f"An error occurred: {str(e)}",
                csv_files=csv_files  # Pass filtered CSV files list
            )

    return render_template('stock_market_prediction.html', csv_files=csv_files)  # Pass filtered CSV files list

@app.route('/download_csv/<filename>')
def download_csv(filename):
    """Serve CSV files for download."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found.", 404


if __name__ == '__main__':
    app.run(debug=True, port=5757)
