import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

# Suppress debug log flooding from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)  # Enables cross-origin resource sharing with your local UI

SEQ_LENGTH = 60

def run_leak_free_lstm(ticker):
    START = "2015-01-01"
    # Automatically tracks historical prices up to the current real-world date
    END = datetime.date.today().strftime("%Y-%m-%d")
    #END = "2026-06-30"
    
    print(f"Initializing Leak-Free Framework for: {ticker}")
    
    # 1. Download data matrix
    df = yf.download(ticker, start=START, end=END)
    df = df.copy()
    
    if df.empty or len(df) < (SEQ_LENGTH + 20):
        raise ValueError(f"Insufficient time-series length found for ticker: {ticker}")

    # 1. Calculate base daily return
    df['Return'] = df['Close'].pct_change()

    # 2. Add Alpha Technical Indicators (Rich Feature Set)
    df['RSI'] = RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    macd_init = MACD(df['Close'].squeeze())
    df['MACD_diff'] = macd_init.macd_diff()
    stoch = StochasticOscillator(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())
    df['Stoch'] = stoch.stoch()

    # 3. Target Variable: Will TOMORROW's close be higher than TODAY's close?
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    # Drop rows with NaNs caused by indicators and the target shift
    df = df.dropna()

    features = ['Close', 'Volume', 'Return', 'RSI', 'MACD_diff', 'Stoch']

    # =========================
    # TEMPORAL SPLIT
    # =========================
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()

    # =========================
    # SCALE FEATURES (ZERO LEAKAGE)
    # =========================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_scaled = scaler.fit_transform(train_df[features])
    test_features_scaled = scaler.transform(test_df[features])

    train_targets = train_df['Target'].values

    # =========================
    # GENERATE SEQUENCES
    # =========================
    def create_binary_sequences(features_scaled, targets, seq_length):
        X, y = [], []
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i])
            y.append(targets[i-1])
        return np.array(X), np.array(y)

    X_train, y_train = create_binary_sequences(train_features_scaled, train_targets, SEQ_LENGTH)

    # =========================
    # ROBUST CLASSIFICATION MODEL
    # =========================
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # =========================
    # TRAINING MECHANICS
    # =========================
    # Set to a highly responsive fitting epoch layout optimized for web API delivery runtimes
    model.fit(
        X_train, y_train,
        epochs=12,
        batch_size=64,
        verbose=0
    )

    # =========================
    # PURE TOMORROW PREDICTION (LIVE OUT OF SAMPLE)
    # =========================
    live_window = test_features_scaled[-SEQ_LENGTH:]
    tomorrow_probability = float(model.predict(live_window.reshape(1, SEQ_LENGTH, len(features)), verbose=0)[0, 0])

    # =========================================================
    # PREDICTED STOCK PRICE VALUE PREPARATION
    # =========================================================
    last_close_price = float(df['Close'].iloc[-1])
    
    avg_up_return = float(train_df.loc[train_df['Return'] > 0, 'Return'].mean())
    avg_down_return = float(train_df.loc[train_df['Return'] < 0, 'Return'].mean())

    predicted_direction = "UP" if tomorrow_probability > 0.5 else "DOWN"
    expected_move = avg_up_return if predicted_direction == "UP" else avg_down_return
    predicted_price = last_close_price * (1 + expected_move)

    return {
        "ticker": ticker,
        "current_price": round(last_close_price, 2),
        "predicted_price": round(predicted_price, 2),
        "direction": predicted_direction
    }

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Missing stock symbol parameter"}), 400
    try:
        results = run_leak_free_lstm(symbol)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)