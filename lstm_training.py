import pandas as pd
import numpy as np
import requests
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

def fetch_data(coin_id='bitcoin', days=200):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    r = requests.get(url, params=params)
    data = r.json()

    prices = data["prices"]
    volumes = data["total_volumes"]

    records = []
    for i in range(len(prices)):
        ts = datetime.utcfromtimestamp(prices[i][0] / 1000)
        close = prices[i][1]
        volume = volumes[i][1]
        records.append({"Date": ts, "Close": close, "Volume": volume})

    df = pd.DataFrame(records)
    df["Open"] = df["Close"].shift(1)
    df["High"] = df["Close"].rolling(2).max()
    df["Low"] = df["Close"].rolling(2).min()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_features(df):
    df = df.copy()
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['ema12'] = df['Close'].ewm(span=12).mean()
    df['ema26'] = df['Close'].ewm(span=26).mean()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df.dropna(inplace=True)
    return df

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main(coin_id):
    print(f"\n📡 Fetching data for {coin_id}...")
    df = fetch_data(coin_id)
    df = prepare_features(df)

    features = ['Open', 'High', 'Low', 'Volume', 'ma7', 'ma21', 'ema12', 'ema26', 'rsi']
    target = 'Close'

    X = df[features].values
    y = df[[target]].values
    
    # ✅ Make sure saving folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, f"models/scaler_X_{coin_id}.save")
    joblib.dump(scaler_y, f"models/scaler_y_close_{coin_id}.save")

    TIME_STEPS = 1
    X_seq, y_seq = create_dataset(X_scaled, y_scaled, TIME_STEPS)
    X_seq = X_seq.reshape((X_seq.shape[0], TIME_STEPS, X_seq.shape[2]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print("🚀 Training model...")
    model.fit(X_seq, y_seq, epochs=100, batch_size=8, verbose=1, callbacks=[es])

    model.save(f"models/lstm_close_model_{coin_id}.h5")
    print(f"✅ Saved: lstm_close_model_{coin_id}.h5")

    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📈 Evaluation for {coin_id}:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # 📊 Visualization Section
    os.makedirs("plots", exist_ok=True)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"{coin_id.upper()} - Actual vs Predicted Close Price")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{coin_id}_actual_vs_predicted.png")
    plt.close()

    # 2. Prediction Error Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_true - y_pred, label='Prediction Error', color='red')
    plt.title(f"{coin_id.upper()} - Prediction Error (Actual - Predicted)")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{coin_id}_prediction_error.png")
    plt.close()

    # 3. Residual Distribution
    plt.figure(figsize=(6, 4))
    residuals = (y_true - y_pred).flatten()
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title(f"{coin_id.upper()} - Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{coin_id}_residual_distribution.png")
    plt.close()

if __name__ == "__main__":
    import sys
    if any('ipykernel' in arg for arg in sys.argv):
        coin_id = 'ethereum'  # change as needed in notebook
        main(coin_id)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--coin', type=str, default='bitcoin', help='CoinGecko coin ID')
        args = parser.parse_args()
        main(args.coin)
