import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crypto Spread Classifier", layout="wide")
st.title("🚀 Crypto Spread Classifier")
st.markdown("Upload raw BTC-USD.csv and ETH-USD.csv to begin.")

btc_file = st.file_uploader("Upload BTC-USD.csv", type="csv")
eth_file = st.file_uploader("Upload ETH-USD.csv", type="csv")

if btc_file and eth_file:
    # 1) Load raw files
    btc_df = pd.read_csv(btc_file, header=0, skiprows=[1])  # skip ticker header row
    eth_df = pd.read_csv(eth_file, header=0, skiprows=[1])

    # 2) Show what we got
    st.subheader("BTC Raw Preview")
    st.dataframe(btc_df.head())
    st.write("Columns:", list(btc_df.columns))
    st.subheader("ETH Raw Preview")
    st.dataframe(eth_df.head())
    st.write("Columns:", list(eth_df.columns))

    # 3) Identify the date column (should be the first column like 'Price' in your file)
    #    and rename it to 'date'
    date_col_btc = btc_df.columns[0]
    date_col_eth = eth_df.columns[0]
    btc_df = btc_df.rename(columns={date_col_btc: 'date'})
    eth_df = eth_df.rename(columns={date_col_eth: 'date'})

    # 4) Parse 'date' and set as index
    btc_df['date'] = pd.to_datetime(btc_df['date'], errors='coerce')
    eth_df['date'] = pd.to_datetime(eth_df['date'], errors='coerce')
    btc_df = btc_df.set_index('date').dropna(subset=['close'])
    eth_df = eth_df.set_index('date').dropna(subset=['close'])

    # 5) Merge on the index
    data = pd.merge(
        btc_df[['close']],
        eth_df[['close']],
        left_index=True,
        right_index=True,
        how='inner'
    ).rename(columns={'close_x': 'bitcoin_close', 'close_y': 'ethereum_close'})

    # 6) Feature engineering
    data['spread'] = data['bitcoin_close'] - data['ethereum_close']
    data['spread_mean'] = data['spread'].rolling(window=30).mean()
    data['spread_std']  = data['spread'].rolling(window=30).std()
    data.dropna(inplace=True)
    data['spread_narrowing'] = np.where(
        data['spread'] < (data['spread_mean'] - data['spread_std']),
        1, 0
    )

    st.subheader("Processed Data Preview")
    st.dataframe(data.tail())

    # 7) Train/test split
    X = data[['spread_mean','spread','ethereum_close','bitcoin_close','spread_std']]
    y = data['spread_narrowing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

    # 8) Pipeline & GridSearch
    preproc = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    X_train_p = preproc.fit_transform(X_train)
    X_test_p  = preproc.transform(X_test)

    st.subheader("Model Training")
    with st.spinner("Running GridSearchCV..."):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth':    [None, 10, 20],
            'criterion':    ['gini','entropy']
        }
        gs = GridSearchCV(RandomForestClassifier(random_state=35), param_grid, cv=5, scoring='accuracy')
        gs.fit(X_train_p, y_train)

    model = gs.best_estimator_
    st.write("Best parameters:", gs.best_params_)

    # 9) Evaluation
    y_pred = model.predict(X_test_p)
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
    st.write(f"Precision: **{precision_score(y_test, y_pred):.4f}**")
    st.write(f"Recall: **{recall_score(y_test, y_pred):.4f}**")
    st.write(f"F1 Score: **{f1_score(y_test, y_pred):.4f}**")
    mse = -cross_val_score(model, X_train_p, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    st.write(f"MSE (CV): **{mse:.4f}**")

    # 10) Training progress plot
    st.subheader("Training Progress")
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
    st.warning("Please upload both CSV files to proceed.")
