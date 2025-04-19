import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("📈 Time Series Forecasting Web App")

# --- Upload File ---
st.sidebar.header("Upload Time Series CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- When File is Uploaded ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    date_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    date_col = st.sidebar.selectbox("Select Date/Time Column", date_cols)
    value_col = st.sidebar.selectbox("Select Value Column", numeric_cols)

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    ts = df[value_col].dropna()
    df = df[[value_col]].dropna()

    # --- Decomposition ---
    st.sidebar.header("Decomposition")
    method = st.sidebar.selectbox("Decomposition Method", ["additive", "multiplicative"])
    st.subheader(f"{method.title()} Decomposition")

    try:
        result = seasonal_decompose(ts, model=method, period=12)
        result.plot()
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Decomposition Error: {e}")

    # --- Forecast Settings ---
    st.sidebar.header("Forecasting")
    forecast_period = st.sidebar.slider("Forecast Period (Months)", 3, 24, 12)
    model_choice = st.sidebar.selectbox("Select Forecasting Model", ["Holt-Winters", "Prophet", "LSTM"])

    st.subheader(f"Forecast using {model_choice}")

    if model_choice == "Holt-Winters":
        try:
            model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12)
            model_fit = model.fit()
            forecast = model_fit.forecast(forecast_period)

            fig, ax = plt.subplots()
            ts.plot(ax=ax, label="Actual")
            forecast.plot(ax=ax, label="Forecast")
            plt.legend()
            st.pyplot(fig)

            # Evaluation
            train = ts[:-12]
            test = ts[-12:]
            eval_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            preds = eval_model.forecast(12)

        except Exception as e:
            st.error(f"Forecasting Error: {e}")

    elif model_choice == "Prophet":
        try:
            prophet_df = df.reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=forecast_period, freq='MS')
            forecast = model.predict(future)

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            test = prophet_df['y'].iloc[-12:]
            preds = forecast['yhat'].iloc[-12:]

        except Exception as e:
            st.error(f"Prophet Forecasting Error: {e}")

    elif model_choice == "LSTM":
        try:
            data = df[[value_col]].values
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            def create_sequences(data, seq_len):
                X, y = [], []
                for i in range(len(data) - seq_len):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len])
                return np.array(X), np.array(y)

            seq_len = 12
            X, y = create_sequences(data_scaled, seq_len)

            X_train, y_train = X[:-forecast_period], y[:-forecast_period]

            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(seq_len, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=30, verbose=0)

            # Forecasting
            input_seq = data_scaled[-seq_len:]
            predictions = []

            for _ in range(forecast_period):
                pred = model.predict(input_seq.reshape(1, seq_len, 1), verbose=0)
                predictions.append(pred[0][0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            # Plot
            fig, ax = plt.subplots()
            ax.plot(ts.index, ts.values, label="Actual")
            future_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='MS')
            ax.plot(future_index, forecast, label="Forecast")
            ax.legend()
            st.pyplot(fig)

            # Evaluation
            test = ts.values[-12:]
            X_test = X[-12:]
            y_pred_scaled = model.predict(X_test, verbose=0)
            preds = scaler.inverse_transform(y_pred_scaled).flatten()

        except Exception as e:
            st.error(f"LSTM Forecasting Error: {e}")

    # --- Evaluation Metrics ---
    try:
        st.subheader("Evaluation Metrics (Last 12 months)")
        mae = mean_absolute_error(test, preds)
        mse = mean_squared_error(test, preds)
        rmse = np.sqrt(mse)

        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
    except:
        st.info("Not enough data to compute evaluation metrics.")