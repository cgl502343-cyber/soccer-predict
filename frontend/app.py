import streamlit as st
import pandas as pd
import pickle
import os

# Load model, scaler, and feature names
with open("model/classifier.pkl", 'rb') as f:
    model = pickle.load(f)
with open("model/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
with open("model/feature_names.pkl", 'rb') as f:
    feature_names = pickle.load(f)

# Since we don't have original team names, just create sample team IDs
# You can customize team names later manually if you want
home_team_ids = [int(col.split('_')[-1]) for col in feature_names if col.startswith('home_team_api_id_')]
away_team_ids = [int(col.split('_')[-1]) for col in feature_names if col.startswith('away_team_api_id_')]
team_ids = list(set(home_team_ids + away_team_ids))
team_dict = {id: f"Team_{id}" for id in team_ids}

# Custom CSS for styling
st.markdown("""
<style>
.main {background-color: #e8f5e9; padding: 20px; border-radius: 10px; font-family: 'Arial', sans-serif;}
.title {color: #2e7d32; font-size: 36px; text-align: center; margin-bottom: 10px;}
.subtitle {color: #388e3c; font-size: 18px; text-align: center; margin-bottom: 30px;}
.input-container {background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
.stNumberInput > div > div > input, .stSelectbox > div > div > select {border: 2px solid #4caf50; border-radius: 5px; padding: 8px; font-size: 16px;}
.predict-button {background-color: #4caf50; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 18px; cursor: pointer; display: block; margin: 0 auto;}
.predict-button:hover {background-color: #45a049;}
.result {color: #d81b60; font-size: 24px; font-weight: bold; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Soccer Match Outcome Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the outcome of a soccer match using pre-match betting odds and team selection</p>', unsafe_allow_html=True)

# Input form
st.markdown('<div class="input-container">', unsafe_allow_html=True)
home_team = st.selectbox("Select Home Team", options=list(team_dict.keys()), format_func=lambda x: team_dict[x], key="home_team")
away_team = st.selectbox("Select Away Team", options=list(team_dict.keys()), format_func=lambda x: team_dict[x], key="away_team")
b365h = st.number_input("Bet365 Home Win Odds", min_value=1.0, value=2.0, step=0.1, key="b365h")
b365d = st.number_input("Bet365 Draw Odds", min_value=1.0, value=3.0, step=0.1, key="b365d")
b365a = st.number_input("Bet365 Away Win Odds", min_value=1.0, value=4.0, step=0.1, key="b365a")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if st.button("Predict", key="predict", help="Click to predict the match outcome"):
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data['B365H'] = b365h
    input_data['B365D'] = b365d
    input_data['B365A'] = b365a
    input_data[f'home_team_api_id_{home_team}'] = 1
    input_data[f'away_team_api_id_{away_team}'] = 1
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    outcome = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    st.markdown(f'<p class="result">Predicted Outcome: {outcome[prediction]}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
