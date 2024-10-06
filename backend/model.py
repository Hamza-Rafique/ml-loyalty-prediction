
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('dataset.csv')

X = data.drop('Loyalty_Score', axis=1)
y = data['Loyalty_Score']

clf = RandomForestClassifier()
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
