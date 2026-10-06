import joblib


# Load trained model
model = joblib.load("model.pkl")

print("======================================")
print("       HOUSE PRICE PREDICTION")
print("       XGBoost Regressor")
print("======================================")

# Get user input
area = float(input("Enter area (sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
floors = int(input("Enter number of floors: "))
age = int(input("Enter house age (years): "))
parking = int(input("Enter parking spaces: "))

# Prepare input
new_house = [[
    area,
    bedrooms,
    bathrooms,
    floors,
    age,
    parking
]]

# Predict
prediction = model.predict(new_house)

# Display result
print("\n======================================")
print("          PREDICTION RESULT")
print("======================================")

print(f"Predicted House Price: ₹{prediction[0]:.2f} Lakhs")

print("======================================")