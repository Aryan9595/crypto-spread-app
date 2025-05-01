import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# 1) Upload Files
st.title("Crypto Price Spread Prediction")

btc_file = st.file_uploader("Upload BTC Data", type="csv")
eth_file = st.file_uploader("Upload ETH Data", type="csv")

if btc_file is not None and eth_file is not None:
    # 2) Load the Data
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    # 3) Clean and process data
    btc["date"] = pd.to_datetime(btc["date"])  # Ensuring date is in datetime format
    eth["date"] = pd.to_datetime(eth["date"])

    df = pd.DataFrame()
    df["BTC_Close"] = btc["Close"]
    df["ETH_Close"] = eth["Close"]
    df["Date"] = btc["date"]  # Assuming both datasets have the same length and same dates

    # 4) Feature Engineering: Create a target for price spread prediction
    df['Spread'] = df['BTC_Close'] - df['ETH_Close']
    df['Target'] = (df['Spread'].shift(-1) > 0).astype(int)  # 1 if next day's spread is positive (buy), 0 if negative (sell)

    # Drop rows with missing target values
    df = df.dropna()

    # 5) Feature selection and target setup
    X = df[["BTC_Close", "ETH_Close"]]
    y = df["Target"]

    # 6) Split Data into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 7) Model training
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 8) Predictions and Metrics
    y_pred = model.predict(X_test)

    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
    st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
    st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
    st.write(f"F1 Score: **{f1_score(y_test, y_pred):.4f}**")

    mse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    st.write(f"MSE (CV): **{mse:.4f}**")

    # 9) Visualization: Plot the BTC and ETH Closing Prices
    st.subheader("BTC and ETH Closing Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Date"], df["BTC_Close"], label="BTC Close")
    ax.plot(df["Date"], df["ETH_Close"], label="ETH Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # 10) Training Progress Plot
    st.subheader("Training Progress")
    train_acc, test_acc = [], []
    for i in range(10, len(X_train), 10):
        model.fit(X_train[:i], y_train[:i])
        train_acc.append(accuracy_score(y_train[:i], model.predict(X_train[:i])))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))

    fig, ax = plt.subplots()
    ax.plot(range(10, len(X_train), 10), train_acc, label="Train")
    ax.plot(range(10, len(X_train), 10), test_acc, label="Test")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    # 11) Make Predictions for Buy and Sell
    st.subheader("Predicted Buy/Sell Signals")
    predicted_actions = pd.DataFrame({
        "Date": df["Date"].iloc[len(X_train):len(X_train) + len(y_test)],
        "Predicted Action": y_pred
    })
    predicted_actions["Predicted Action"] = predicted_actions["Predicted Action"].map({0: "Sell", 1: "Buy"})
    st.write(predicted_actions)

else:
    st.warning("Please upload both CSV files to proceed.")
