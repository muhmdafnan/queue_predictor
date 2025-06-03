import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load and train model
@st.cache_resource
def load_model_and_scaler():
    df = pd.read_excel('3 PAGI 3_updated.xlsx', engine='openpyxl')
    df['Hour'] = df['Visit Date'].dt.hour
    df['Minute'] = df['Visit Date'].dt.minute
    df['DayOfWeek'] = df['Visit Date'].dt.dayofweek
    df['IsPeak'] = df['Time Peak'].apply(lambda x: 1 if x == 'Peak Time' else 0)
    
    X = df[['Hour', 'Minute', 'DayOfWeek', 'IsPeak', 'Service Duration Minutes', 'Queue Length']]
    y = df['Queue Waiting Minutes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_model_and_scaler()

# UI
st.title("⏳ Queue Waiting Time Predictor")
st.caption("Predict how long someone will wait in line.")

# Inputs
hour = st.slider("Hour of Visit", 7, 17, 10)
minute = st.slider("Minute of Visit", 0, 59, 30)

day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
day = st.selectbox("Day of the Week", list(day_map.keys()))
dayofweek = day_map[day]

is_peak = st.radio("Is it Peak Time?", ['Yes', 'No'])
is_peak_num = 1 if is_peak == 'Yes' else 0

service_duration = st.slider("Service Duration (minutes)", 1, 30, 10)
queue_length = st.slider("Queue Length", 0, 20, 5)

# Predict
if st.button("Predict"):
    features = np.array([[hour, minute, dayofweek, is_peak_num, service_duration, queue_length]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    lower = max(0, round(prediction - 2))
    upper = round(prediction + 2)

    st.subheader(f"🎯 Estimated Wait Time: {round(prediction)} minutes")
    st.caption(f"📊 Confidence Range: {lower} – {upper} minutes")
