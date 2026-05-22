"""
train.py
--------
Production training script for NHS A&E Breach Predictor.
Trains Random Forest on full processed dataset and saves
the model to models/random_forest.pkl.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data/processed/nhs_processed.csv')

X = df[['type1_att', 'admission_rate', 'severe_wait_ratio']]
y = df['breached']

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved to models/random_forest_model.pkl')