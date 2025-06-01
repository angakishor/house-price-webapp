# house_price_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Generate synthetic dataset with 1000+ samples
np.random.seed(42)
num_samples = 1200

data = pd.DataFrame({
    'area': np.random.randint(800, 3500, num_samples),
    'size': np.random.randint(50, 300, num_samples),
    'rooms': np.random.randint(1, 7, num_samples),
    'type': np.random.choice(['individual', 'flat'], num_samples)
})

# Generate synthetic prices based on a rough formula + noise
data['price'] = (
    data['area'] * 0.05 +
    data['size'] * 0.8 +
    data['rooms'] * 10 +
    np.where(data['type'] == 'individual', 50, 0) +
    np.random.normal(0, 10, num_samples)
).round(2)

# Encode categorical variable
data['type'] = data['type'].map({'individual': 1, 'flat': 0})

# Features and Target
X = data[['area', 'size', 'rooms', 'type']]
y = data['price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.2f}")

# Save Model
joblib.dump(model, 'house_price_model.pkl')

