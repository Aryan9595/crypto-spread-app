import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Page Config ---
st.set_page_config(page_title="Crypto Spread Classifier", layout="wide")
st.title("📈 Crypto Spread Classifier & Buy/Sell Predictor")

# --- File Upload ---
btc_file = st.file_uploader("Upload BTC CSV", type="csv")
eth_file = st.file_uploader("Upload ETH CSV", type="csv")

if btc_file and eth_file:
    # --- Load & Normalize Columns ---
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)
    btc.columns = btc.columns.str.lower().str.strip()
    eth.columns = eth.columns.str.lower().str.strip()
    
    # --- Parse Dates Safely ---
    btc["date"] = pd.to_datetime(btc["date"], format="%Y-%m-%d", errors="coerce")
    eth["date"] = pd.to_datetime(eth["date"], format="%Y-%m-%d", errors="coerce")
    btc = btc.dropna(subset=["date"])
    eth = eth.dropna(subset=["date"])
    
    # --- Merge on Date ---
    df = pd.merge(btc, eth, on="date", suffixes=("_btc", "_eth"))
    df = df.sort_values("date").reset_index(drop=True)
    
    # --- Feature Engineering ---
    df["spread"] = df["close_btc"] - df["close_eth"]
    # Signal: 1 = spread goes up next day (Buy), 0 = goes down or flat (Sell)
    df["signal"] = (df["spread"].shift(-1) > df["spread"]).astype(int)
    df = df.dropna(subset=["signal"])
    
    # --- Prepare Data ---
    X = df[["close_btc", "close_eth", "spread"]]
    y = df["signal"]
    
    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # --- Model Training ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # --- Metrics ---
    st.subheader("📊 Model Performance")
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    mse   = -cross_val_score(
                model, X_train, y_train,
                cv=5, scoring="neg_mean_squared_error"
            ).mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",    f"{acc:.4f}")
    c2.metric("Precision",   f"{prec:.4f}")
    c3.metric("Recall",      f"{rec:.4f}")
    c4.metric("F1 Score",    f"{f1:.4f}")
    st.metric("MSE (CV)", f"{mse:.4f}")
    
    # --- Training Progress Plot ---
    st.subheader("📈 Training Progress")
    train_acc, test_acc, steps = [], [], []
    for i in range(10, len(X_train), 10):
        model.fit(X_train.iloc[:i], y_train.iloc[:i])
        train_acc.append(accuracy_score(
            y_train.iloc[:i], model.predict(X_train.iloc[:i])
        ))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))
        steps.append(i)
    
    fig, ax = plt.subplots()
    ax.plot(steps, train_acc, label="Train")
    ax.plot(steps, test_acc, label="Test")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # --- Buy/Sell Predictions Table ---
    st.subheader("📌 Buy/Sell Predictions")
    pred_df = pd.DataFrame({
        "Date": df["date"].iloc[-len(y_test):].dt.strftime("%Y-%m-%d"),
        "BTC_Close": df["close_btc"].iloc[-len(y_test):].values,
        "ETH_Close": df["close_eth"].iloc[-len(y_test):].values,
        "Spread": df["spread"].iloc[-len(y_test):].values,
        "Prediction": y_pred
    })
    pred_df["Action"] = pred_df["Prediction"].map({1: "Buy", 0: "Sell"})
    st.dataframe(pred_df[["Date", "BTC_Close", "ETH_Close", "Spread", "Action"]])

    # --- Download Button ---
    csv = pred_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Predictions",
        csv,
        file_name="crypto_spread_predictions.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Please upload both BTC and ETH CSV files to proceed.")
