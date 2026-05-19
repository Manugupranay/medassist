import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

np.random.seed(42)
n = 2000
X = np.random.randn(n, 9)
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X, y)

os.makedirs('models', exist_ok=True)
with open('models/risk_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print('Model saved!')
