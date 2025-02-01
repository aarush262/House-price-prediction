import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LinearRegression

# Create Flask app
app = Flask(__name__)

# Sample dataset (or use real data)
data = {
    "Square_Feet": [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5, 6],
    "Location": [1, 2, 2, 3, 3, 4, 4, 5],  # Encoded values for locations
    "Price": [300000, 350000, 500000, 600000, 650000, 700000, 750000, 850000]
}

df = pd.DataFrame(data)

# Train the model
X = df[["Square_Feet", "Bedrooms", "Location"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# Save the model using pickle
pickle.dump(model, open("model.pkl", "wb"))

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    square_feet = float(request.form['square_feet'])
    bedrooms = int(request.form['bedrooms'])
    location = int(request.form['location'])

    # Make prediction
    input_data = np.array([[square_feet, bedrooms, location]])
    predicted_price = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Estimated House Price: ${predicted_price:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
