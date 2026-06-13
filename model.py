import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_csv("train (1).csv")

# Select important features
X = df[['LotArea', 'OverallQual', 'YearBuilt',
        'BedroomAbvGr', 'FullBath', 'GarageCars']]

# Target column
y = df['SalePrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
pred = model.predict(X_test)

# Calculate accuracy
accuracy = r2_score(y_test, pred) * 100

print(f"Model Accuracy: {accuracy:.2f}%")

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")