import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# GLOBAL SESSION (FAST API)
# =========================
session = requests.Session()

def safe_request(url, method="GET", **kwargs):
    try:
        if method == "GET":
            resp = session.get(url, timeout=10, **kwargs)
        else:
            resp = session.post(url, timeout=15, **kwargs)

        resp.raise_for_status()

        try:
            return resp.json()
        except:
            return None

    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None


# =========================
# LOCATION FUNCTIONS
# =========================
@st.cache_data(ttl=86400)
def fetch_ip_location():
    data = safe_request("http://ip-api.com/json/")
    if not data:
        return {"lat": 20.5937, "lon": 78.9629}
    return {
        "lat": data.get("lat", 20.5937),
        "lon": data.get("lon", 78.9629),
    }


@st.cache_data(ttl=86400)
def geocode_place(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    data = safe_request(url, params=params)

    if not data:
        return None

    return float(data[0]["lat"]), float(data[0]["lon"])


# =========================
# WEATHER
# =========================
@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "forecast_days": 3,
    }

    data = safe_request(url, params=params)

    if not data or "hourly" not in data:
        return pd.DataFrame()

    h = data["hourly"]

    df = pd.DataFrame({
        "temp": h["temperature_2m"],
        "rain": h["precipitation"],
        "wind": h["windspeed_10m"],
    })

    return df


# =========================
# MODEL (TRAIN ONCE)
# =========================
@st.cache_resource
def load_model():
    model = RandomForestClassifier()
    scaler = StandardScaler()

    # synthetic data
    rain = np.random.uniform(0, 150, 1000)
    wind = np.random.uniform(0, 120, 1000)
    temp = np.random.uniform(10, 45, 1000)

    score = rain*0.6 + wind*0.3 + np.maximum(temp-30, 0)*0.2
    y = pd.cut(score, bins=[-1, 30, 80, 999], labels=[0,1,2]).astype(int)

    X = pd.DataFrame({
        "rain": rain,
        "wind": wind,
        "temp": temp
    })

    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    return model, scaler


def predict(model, scaler, rain, wind, temp):
    X = pd.DataFrame([{"rain": rain, "wind": wind, "temp": temp}])
    Xs = scaler.transform(X)

    pred = int(model.predict(Xs)[0])
    prob = np.max(model.predict_proba(Xs))

    return pred, prob


def risk_label(r):
    return ["LOW", "MEDIUM", "HIGH"][r]


# =========================
# UI
# =========================
st.title("🚨 AI Disaster Response System")

mode = st.radio("Mode", ["Auto", "Manual"])

if mode == "Auto":
    loc = fetch_ip_location()
    lat, lon = loc["lat"], loc["lon"]
else:
    place = st.text_input("Enter location")
    coords = geocode_place(place)
    if coords:
        lat, lon = coords
    else:
        lat, lon = 20.5937, 78.9629

st.write(f"📍 Location: {lat:.2f}, {lon:.2f}")

# =========================
# WEATHER + MODEL
# =========================
weather = fetch_weather(lat, lon)

if weather.empty:
    st.error("Weather data unavailable")
    st.stop()

rain = weather["rain"].sum()
wind = weather["wind"].max()
temp = weather["temp"].mean()

model, scaler = load_model()

risk, conf = predict(model, scaler, rain, wind, temp)

# =========================
# OUTPUT
# =========================
st.metric("Rainfall", f"{rain:.1f} mm")
st.metric("Wind", f"{wind:.1f} km/h")
st.metric("Temperature", f"{temp:.1f} °C")

st.subheader(f"Risk Level: {risk_label(risk)}")

if risk == 2:
    st.error("⚠️ High Risk! Take precautions.")
elif risk == 1:
    st.warning("⚠️ Moderate Risk.")
else:
    st.success("✅ Low Risk")

st.caption(f"Confidence: {conf*100:.1f}%")