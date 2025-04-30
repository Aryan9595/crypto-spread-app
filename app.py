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

st.title("📈 Crypto Spread Classifier")
st.markdown("Upload BTC-USD and ETH-USD CSV files to begin.")

btc_file = st.file_uploader("Upload BTC-USD.csv", type=["csv"])
eth_file = st.file_uploader("Upload ETH-USD.csv", type=["csv"])

if btc_file and eth_file:
    btc = pd.read_csv(btc_file, index_col="Date", parse_dates=True)
    eth = pd.read_csv(eth_file, index_col="Date", parse_dates=True)

    data = pd.merge(btc[['Close']], eth[['Close']], on="Date")
    data.columns = ['Bitcoin_Close', 'Ethereum_Close']

    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed['Spread'] = data_imputed['Bitcoin_Close'] - data_imputed['Ethereum_Close']
    data_imputed['Spread_mean'] = data_imputed['Spread'].rolling(window=30).mean()
    data_imputed['Spread_std'] = data_imputed['Spread'].rolling(window=30).std()
    data_imputed['Spread_Narrowing'] = np.where(
        data_imputed['Spread'] < (data_imputed['Spread_mean'] - data_imputed['Spread_std']), 1, 0
    )

    data_imputed.dropna(inplace=True)

    st.subheader("Data Preview")
    st.write(data_imputed.tail())

    features = ['Spread_mean', 'Spread', 'Ethereum_Close', 'Bitcoin_Close', 'Spread_std']
    X = data_imputed[features]
    y = data_imputed['Spread_Narrowing']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
    preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=35), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_p, y_train)
    best_params = grid_search.best_params_
    model = RandomForestClassifier(**best_params, random_state=35)
    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mse = cross_val_score(model, X_train_p, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    cross_val = cross_val_score(model, X_train_p, y_train, cv=4).mean()

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**Mean Squared Error (CV):** {mse:.4f}")
    st.write(f"**Cross-validation Score:** {cross_val:.4f}")

    st.subheader("📉 Training Progress Visualization")
    train_acc = []
    test_acc = []
    for i in range(10, len(X_train_p)):
        model.fit(X_train_p[:i], y_train[:i])
        train_acc.append(accuracy_score(y_train[:i], model.predict(X_train_p[:i])))
        test_acc.append(accuracy_score(y_test, model.predict(X_test_p)))

    fig, ax = plt.subplots()
    ax.plot(range(10, len(X_train_p)), train_acc, label="Train Acc")
    ax.plot(range(10, len(X_train_p)), test_acc, label="Test Acc")
    ax.legend()
    ax.set_title("Accuracy over training")
    st.pyplot(fig)

    st.subheader("📈 Risk Metrics")
    closing = data_imputed[['Bitcoin_Close', 'Ethereum_Close']]
    returns = closing.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()).mean()
    sortino_ratio = (
        returns[returns > 0].mean() /
        returns[returns < 0].std()
    ).mean()
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.4f}")
    st.write(f"**Sortino Ratio:** {sortino_ratio:.4f}")

    st.subheader("💼 Trade Simulation")
    trade_size = st.slider("Trade Size", min_value=10, max_value=1000, value=100)
    fee = st.slider("Transaction Fee %", 0.0, 0.1, 0.01)
    slippage = st.slider("Market Impact %", 0.0, 0.1, 0.02)

    st.write("**Simulated Trades (First 10):**")
    trades = []
    for pred in y_pred[:10]:
        total_cost = trade_size * fee + trade_size * slippage
        final_trade_size = trade_size - total_cost
        trades.append({
            "Signal": "BUY" if pred == 1 else "SELL",
            "Final Size": round(final_trade_size, 2)
        })
    st.dataframe(pd.DataFrame(trades))
else:
    st.warning("Upload both BTC-USD and ETH-USD CSV files to continue.")
