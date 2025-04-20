import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from openai import OpenAI
from sklearn.model_selection import cross_val_score


client= OpenAI(api_key="")

def fetch_binance_ohlc():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "LTCUSDT",
        "interval": "1d",
        "limit": 1500
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    ohlc = []
    for item in data:
        date = datetime.fromtimestamp(item[0] / 1000).date()
        open_price = float(item[1])
        high_price = float(item[2])
        low_price = float(item[3])
        close_price = float(item[4])
        volume = float(item[5])
        ohlc.append([date, open_price, high_price, low_price, close_price, volume])

    df = pd.DataFrame(ohlc, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return df

def cross_validate_model(model, X_scaled, y):
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print("\n🧪 Cross-validation results (R²):", scores)
    print(f"📊 Mean R²: {scores.mean():.4f}")

def train_neural_network():
    print("📊 Loading information from Binance...")
    df = fetch_binance_ohlc()

    # Feature və target ayır
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(hidden_layer_sizes=(128, 128),
                         activation='relu',
                         solver='adam',
                         max_iter=10000,
                         random_state=42)

    print("🚀 Trainin the model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    cross_validate_model(model, X_scaled, y)

    print(f"🎯 R² Score: {r2:.4f}")

    return model, scaler, df

def predict_price(model, scaler, df):
    year = int(input("Input year (example: 2025): "))
    month = int(input("Input month (1-12): "))
    day = int(input("Input day (1-31): "))

    try:
        selected_date = datetime(year, month, day).date()
        row = df[df["Date"] == selected_date]

        if row.empty:
            print("❌ There is no information for this date")
            return


        input_features = row[["Open", "High", "Low", "Volume"]].values
        input_scaled = scaler.transform(input_features)
        predicted_price = model.predict(input_scaled)[0]

        open_price = row["Open"].values[0]
        high_price = row["High"].values[0]
        low_price = row["Low"].values[0]
        volume = row["Volume"].values[0]

        print(f"\n📅 Date: {selected_date}")
        print(f"🤖 Predicted Price: ${predicted_price:.4f}")
        print(f"✅ Actual Price: ${row['Close'].values[0]:.4f}")
        print("\n🧠 ChatGPT Explanation:")
        explain_prediction(open_price, high_price, low_price, volume, predicted_price)


    except Exception as e:
        print(f"Error: {e}")



def explain_prediction(open_price, high_price, low_price, volume, predicted_price):
    prompt = (
        f"Open: ${open_price:.2f}\n"
        f"High: ${high_price:.2f}\n"
        f"Low: ${low_price:.2f}\n"
        f"Volume: {volume:.2f}\n"
        f"Predicted Price: ${predicted_price:.2f}\n\n"
        f"Please explain this result."
    )

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
    )

    explanation = response.choices[0].message.content
    print(explanation)


if __name__ == "__main__":
    model, scaler, df = train_neural_network()
    predict_price(model, scaler, df)

