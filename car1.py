import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
file_path = 'laptopPrice.csv'
df = pd.read_csv(file_path)

# Preprocess the data: handle missing values if any
data = data.dropna()

# Encode categorical features
label_encoder = LabelEncoder()
data['fuel'] = label_encoder.fit_transform(data['fuel'])
data['seller_type'] = label_encoder.fit_transform(data['seller_type'])
data['transmission'] = label_encoder.fit_transform(data['transmission'])
data['owner'] = label_encoder.fit_transform(data['owner'])

# Select features and target variable
X = data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = data['selling_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'Root Mean Squared Error: {rmse}')

# Save the model
joblib.dump(model, 'car_price_model.pkl')

# Function to make predictions with the trained model
def predict_car_price(input_data):
    # Convert categorical features using the same label encoders
    input_data['fuel'] = label_encoder.fit_transform(input_data['fuel'])
    input_data['seller_type'] = label_encoder.fit_transform(input_data['seller_type'])
    input_data['transmission'] = label_encoder.fit_transform(input_data['transmission'])
    input_data['owner'] = label_encoder.fit_transform(input_data['owner'])

    # Predict the car price
    prediction = model.predict(input_data)
    return prediction

# Example usage of the prediction function
example_data = pd.DataFrame({
    'year': [2015],
    'km_driven': [50000],
    'fuel': ['Petrol'],
    'seller_type': ['Individual'],
    'transmission': ['Manual'],
    'owner': ['First Owner']
})
predicted_price = predict_car_price(example_data)
print(f'Predicted Selling Price: {predicted_price[0]}')
