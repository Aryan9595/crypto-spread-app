import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Crypto Spread Analyzer", layout="wide")

st.title("📈 Crypto Spread ML App")
st.write("Upload BTC and ETH CSVs to analyze price spread and apply ML modeling.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload & Preview", "📊 Spread Visualization", "🤖 Train Model"])

with tab1:
    st.header("Step 1: Upload CSV files")
    btc_file = st.file_uploader("Upload Bitcoin (BTC) CSV", type=["csv"])
    eth_file = st.file_uploader("Upload Ethereum (ETH) CSV", type=["csv"])

    if btc_file and eth_file:
        btc = pd.read_csv(btc_file, index_col=0, parse_dates=True)
        eth = pd.read_csv(eth_file, index_col=0, parse_dates=True)

        st.success("Files uploaded successfully.")
        st.subheader("📄 BTC Preview")
        st.dataframe(btc.head())
        st.subheader("📄 ETH Preview")
        st.dataframe(eth.head())
    else:
        st.warning("Please upload both BTC and ETH CSV files to continue.")

if btc_file and eth_file:
    # Preprocessing
    df = pd.DataFrame()
    df["BTC_Close"] = btc["Close"]
    df["ETH_Close"] = eth["Close"]
    df.dropna(inplace=True)

    # Spread calculation
    df["Spread"] = df["BTC_Close"] - df["ETH_Close"]
    df["Signal"] = df["Spread"].diff().apply(lambda x: 1 if x > 0 else 0)

    with tab2:
        st.header("Step 2: Visualize Spread")

        st.line_chart(df[["BTC_Close", "ETH_Close"]])
        st.subheader("Spread")
        st.line_chart(df["Spread"])

    with tab3:
        st.header("Step 3: Train ML Model")
        
        X = df[["BTC_Close", "ETH_Close"]]
        y = df["Signal"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Scale
        scaler = StandardScaler()
        X_train_p = scaler.fit_transform(X_train)
        X_test_p = scaler.transform(X_test)

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)

        # Metrics
        st.subheader("📊 Evaluation Metrics")
        st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
        st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
        st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
        st.write(f"F1 Score: **{f1_score(y_test, y_pred):.4f}**")

        mse = -cross_val_score(model, X_train_p, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        st.write(f"MSE (CV): **{mse:.4f}**")

        # Progress plot
        st.subheader("📈 Training Progress Over Time")
        train_acc, test_acc = [], []
        for i in range(10, len(X_train_p), 10):
            model.fit(X_train_p[:i], y_train[:i])
            train_acc.append(accuracy_score(y_train[:i], model.predict(X_train_p[:i])))
            test_acc.append(accuracy_score(y_test, model.predict(X_test_p)))

        fig, ax = plt.subplots()
        ax.plot(range(10, len(X_train_p), 10), train_acc, label="Train Accuracy")
        ax.plot(range(10, len(X_train_p), 10), test_acc, label="Test Accuracy")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

else:
    with tab2:
        st.warning("Upload files in 'Upload & Preview' tab to see visualizations.")
    with tab3:
        st.warning("Upload files in 'Upload & Preview' tab to train the model.")
