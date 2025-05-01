import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")

st.title("📈 Crypto Spread Analyzer and Predictor")

btc_file = st.file_uploader("Upload BTC CSV", type=["csv"])
eth_file = st.file_uploader("Upload ETH CSV", type=["csv"])

if btc_file and eth_file:
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    for df in [btc, eth]:
        df.columns = df.columns.str.lower().str.strip()

    btc["date"] = pd.to_datetime(btc["date"], errors="coerce")
    eth["date"] = pd.to_datetime(eth["date"], errors="coerce")

    df = pd.merge(btc, eth, on="date", suffixes=("_btc", "_eth"))
    df["spread"] = df["close_btc"] - df["close_eth"]

    df["target"] = (df["spread"].shift(-1) > df["spread"]).astype(int)
    df.dropna(inplace=True)

    X = df[["open_btc", "high_btc", "low_btc", "close_btc", "volume_btc",
            "open_eth", "high_eth", "low_eth", "close_eth", "volume_eth", "spread"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(X_train)
    X_test_p = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    st.subheader("📊 Model Evaluation")
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
    st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
    st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
    st.write(f"F1 Score: **{f1_score(y_test, y_pred):.4f}**")
    mse = -cross_val_score(model, X_train_p, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    st.write(f"MSE (CV): **{mse:.4f}**")

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

    st.subheader("📌 Buy/Sell Recommendations")
    test_dates = df.iloc[y_test.index]["date"]
    prices = df.iloc[y_test.index]["close_btc"]
    buy_sell = ["Buy" if pred == 1 else "Sell" for pred in y_pred]

    recommendation_df = pd.DataFrame({
        "Date": test_dates.values,
        "BTC Price": prices.values,
        "Prediction": buy_sell
    }).reset_index(drop=True)

    st.dataframe(recommendation_df)

    st.download_button("📥 Download Predictions CSV", data=recommendation_df.to_csv(index=False),
                       file_name="crypto_predictions.csv", mime="text/csv")

else:
    st.warning("Please upload both CSV files to proceed.")
