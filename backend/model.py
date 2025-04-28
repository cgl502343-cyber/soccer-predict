import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import os

def load_data():
    # Just load directly using relative path
    data_path = os.path.join("dataset", "soccer_data.csv")  
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    features = ['B365H', 'B365D', 'B365A', 'home_team_api_id', 'away_team_api_id']
    df = df[features + ['home_team_goal', 'away_team_goal']].dropna()
    df['match_result'] = np.where(df['home_team_goal'] > df['away_team_goal'], 0,
                                  np.where(df['home_team_goal'] == df['away_team_goal'], 1, 2))
    df = pd.get_dummies(df, columns=['home_team_api_id', 'away_team_api_id'])
    X = df.drop(['match_result', 'home_team_goal', 'away_team_goal'], axis=1)
    y = df['match_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_model(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

    final_model = lr_model if lr_accuracy > nb_accuracy else nb_model
    print(f"Selected model accuracy: {max(lr_accuracy, nb_accuracy):.2f}")
    return final_model

def save_model(model, scaler, feature_names):
    os.makedirs("model", exist_ok=True)  # create model folder if not exists
    with open("model/classifier.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("model/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open("model/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model, scaler, feature_names)
