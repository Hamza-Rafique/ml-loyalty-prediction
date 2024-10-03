
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('dataset.csv')

# Features and target
X = data.drop('Loyalty_Score', axis=1)
y = data['Loyalty_Score']

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save model to disk
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
