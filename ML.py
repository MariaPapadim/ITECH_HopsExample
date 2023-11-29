import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib 

# Load JSON data from file
file_path = 'points.json'
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Convert lists to numpy arrays
X = np.array(data['X']).reshape(-1, 1)  # Reshape to a column vector
Y = np.array(data['Y']).reshape(-1, 1)
Z = np.array(data['Z']).reshape(-1, 1)

# Stack X and Y arrays horizontally to form the input feature array
X_train = np.column_stack((X, Y))

# Use Z as the target output array
y_train = Z

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a Linear Regression model
#model = LinearRegression()
params = {
"n_estimators": 500,
"max_depth": 4,
"min_samples_split": 5,
"learning_rate": 0.01,
"loss": "squared_error",
}
model = GradientBoostingRegressor(**params)

# Train the model
model.fit(X_train, y_train)


# Save the trained model to a file
model_filename = 'gradient_boosting_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to: {model_filename}")


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Predict for new values
new_data = np.array([[233.989,2.7217]])
new_prediction = model.predict(new_data)
print("Prediction for new data:", new_prediction)