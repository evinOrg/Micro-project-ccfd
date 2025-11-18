# app.py
# ------------------------------------------------------------------
# 0.  IMPORTS
# ------------------------------------------------------------------
import pathlib
import joblib
import pandas as pd
import streamlit as st
import datetime
import numpy
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# 1.  CONSTANTS
# ------------------------------------------------------------------
MODEL_DIR = pathlib.Path(".")          # folder with artefacts
CAT_COLS = ['job', 'category', 'merchant', 'time_of_day']
MODEL_FEATURES = [
    'amt', 'job', 'category', 'merchant',
    'lat', 'long', 'merch_lat', 'merch_long',
    'hour', 'day', 'month', 'day_of_week', 'time_of_day'
]

# ------------------------------------------------------------------
# 2.  BUILD LABEL ENCODERS *ONCE* (same data / logic as training)
# ------------------------------------------------------------------


@st.cache_data(show_spinner=False)   # cache so it runs only once
def _build_encoders():
    df = pd.read_csv("fraudtest.csv")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    def _time_of_day(h: int) -> str:
        if h < 6:
            return 'night'
        elif h < 12:
            return 'morning'
        elif h < 18:
            return 'afternoon'
        else:
            return 'evening'

    encoders = {}
    for col in CAT_COLS:
        if col == 'time_of_day':
            ser = df['trans_date_trans_time'].dt.hour.map(_time_of_day)
        else:
            ser = df[col].astype(str)
        le = LabelEncoder()
        le.fit(ser)
        encoders[col] = le
    return encoders


encoders = _build_encoders()

# ------------------------------------------------------------------
# 3.  LOAD MODELS
# ------------------------------------------------------------------
xgb_model = joblib.load(MODEL_DIR / "xgboost_fraud_model.pkl")
rf_model = joblib.load(MODEL_DIR / "random_forest_fraud_model.pkl")

# ------------------------------------------------------------------
# 4.  PRE-PROCESSING FUNCTION
# ------------------------------------------------------------------


def _time_of_day(h: int) -> str:
    if h < 6:
        return 'night'
    elif h < 12:
        return 'morning'
    elif h < 18:
        return 'afternoon'
    else:
        return 'evening'


def preprocess_row(raw_row: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(raw_row['trans_date_trans_time'])
    hour = dt.hour
    day = dt.day
    month = dt.month
    day_of_week = dt.dayofweek
    time_of_day = _time_of_day(hour)

    row = {
        'amt': raw_row['amt'],
        'lat': raw_row['lat'],
        'long': raw_row['long'],
        'merch_lat': raw_row['merch_lat'],
        'merch_long': raw_row['merch_long'],
        'hour': hour,
        'day': day,
        'month': month,
        'day_of_week': day_of_week,
        'time_of_day': time_of_day,
        'job': str(raw_row['job']),
        'category': str(raw_row['category']),
        'merchant': str(raw_row['merchant']),
    }

    # encode categoricals with the *same* encoders
    for col in CAT_COLS:
        row[col] = int(encoders[col].transform([row[col]])[0])

    return pd.DataFrame([row])[MODEL_FEATURES]


# ------------------------------------------------------------------
# 5.  STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("Credit-Card Fraud Detection – Real-Time Demo")

# load lookup table once


@st.cache_data
def load_lookup():
    df = pd.read_csv("fraudtest.csv")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    return df


df = load_lookup()

# user inputs
cc_num = st.text_input("Enter Credit Card Number (cc_num)")
trans_date = st.date_input("Transaction Date")
trans_time_str = st.text_input("Transaction Time (HH:MM:SS)")

if st.button("Search & Predict"):
    if not cc_num:
        st.error("Please enter cc_num")
        st.stop()

    try:
        trans_time = datetime.datetime.strptime(
            trans_time_str, "%H:%M:%S").time()
        full_dt = datetime.datetime.combine(trans_date, trans_time)
        dt_string = full_dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        st.error("Invalid time format! Use HH:MM:SS")
        st.stop()

    match = df[(df.cc_num.astype(str) == cc_num.strip()) &
               (df.trans_date_trans_time == dt_string)]

    if match.empty:
        st.error("❌ No matching record found for cc_num + datetime")
        st.stop()

    st.success("Record found!")
    st.dataframe(match)

    # ----  prediction  ----
    X = preprocess_row(match.iloc[0])
    xgb_pred = int(xgb_model.predict(X)[0])
    xgb_prob = float(xgb_model.predict_proba(X)[0, 1])

    rf_pred = int(rf_model.predict(X)[0])
    rf_prob = float(rf_model.predict_proba(X)[0, 1])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### XGBoost")
        st.write(
            f"Prediction: **{'Fraud' if xgb_pred == 1 else 'Not Fraud'}**")
        st.write(f"Probability: **{xgb_prob:.4f}**")
    with col2:
        st.write("### Random Forest")
        st.write(f"Prediction: **{'Fraud' if rf_pred == 1 else 'Not Fraud'}**")
        st.write(f"Probability: **{rf_prob:.4f}**")

    st.success("Prediction completed")
