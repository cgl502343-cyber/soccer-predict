Soccer Match Outcome Prediction
   A data science mini-project to predict European soccer match outcomes using machine learning.
Project Structure

backend/: Contains model.py for data preprocessing and model training.
frontend/: Contains app.py for the Streamlit app with HTML/CSS styling.
model/: Stores the trained model, scaler, and feature names (classifier.pkl, scaler.pkl, feature_names.pkl).
dataset/: Contains the dataset (soccer_data.csv).

Features

Predicts match outcomes (Home Win, Draw, Away Win) using Logistic Regression or Naive Bayes.
Uses pre-match features: Bet365 betting odds and team IDs.
Attractive UI with custom HTML/CSS styling in Streamlit.
Deployed on Streamlit Community Cloud.

Setup

Clone the repository:git clone <repository-url>


Install dependencies:pip install -r requirements.txt


Run the Streamlit app:streamlit run frontend/app.py



Dataset

European Soccer Database from Kaggle: https://www.kaggle.com/datasets/hugomathien/soccer
Uses the Match table exported as soccer_data.csv.

Inputs for Prediction

Home Team: Selected from a dropdown of team IDs.
Away Team: Selected from a dropdown of team IDs.
Bet365 Home Win Odds: Odds for a home win (e.g., 2.0).
Bet365 Draw Odds: Odds for a draw (e.g., 3.0).
Bet365 Away Win Odds: Odds for an away win (e.g., 4.0).

Deployment

Deployed on Streamlit Community Cloud: [Link to be added after deployment]

