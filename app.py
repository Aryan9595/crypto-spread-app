import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")
st.title("📈 Crypto Price Spread Classifier (BTC vs ETH)")

btc_file = st.file_uploader("Upload BTC CSV", type=["csv"])
eth_file = st.file_uploader("Upload ETH CSV", type=["csv"])

if btc_file and eth_file:
    # 1) Load data
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    # 2) Preprocessing: lowercase columns and parse date
    btc.columns = btc.columns.str.lower()
    eth.columns = eth.columns.str.lower()
    btc["date"] = pd.to_datetime(btc["date"])
    eth["date"] = pd.to_datetime(eth["date"])
    btc.set_index("date", inplace=True)
    eth.set_index("date", inplace=True)

    # 3) Merge data
    df = pd.DataFrame()
    df["btc_close"] = btc["close"]
    df["eth_close"] = eth["close"]
    df.dropna(inplace=True)

    # 4) Calculate spread and label
    df["spread"] = df["btc_close"] - df["eth_close"]
    threshold = df["spread"].median()
    df["target"] = (df["spread"] > threshold).astype(int)

    # 5) Features
    df["btc_pct"] = btc["close"].pct_change()
    df["eth_pct"] = eth["close"].pct_change()
    df.dropna(inplace=True)

    X = df[["btc_pct", "eth_pct"]]
    y = df["target"]

    # 6) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 7) Scaling
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(X_train)
    X_test_p = scaler.transform(X_test)

    # 8) Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    # 9) Metrics
    st.subheader("📊 Model Performance")
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
    st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
    st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
    st.write(f"F1 Score: **{f1_score(y_test, y_pred):.4f}**")
    mse = -cross_val_score(model, X_train_p, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    st.write(f"MSE (CV): **{mse:.4f}**")

    # 10) Training progress plot
    st.subheader("📈 Training Progress")
    train_acc, test_acc = [], []
    for i in range(10, len(X_train_p), 10):
        model.fit(X_train_p[:i], y_train[:i])
        train_acc.append(accuracy_score(y_train[:i], model.predict(X_train_p[:i])))
        test_acc.append(accuracy_score(y_test, model.predict(X_test_p)))

    fig, ax = plt.subplots()
    ax.plot(range(10, len(X_train_p), 10), train_acc, label="Train")
    ax.plot(range(10, len(X_train_p), 10), test_acc, label="Test")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload both CSV files to proceed.")
