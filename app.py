import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        city = request.form['city']
        property_type = request.form['propertytype']

        lotarea = float(request.form['lotarea'])
        overallqual = float(request.form['overallqual'])
        yearbuilt = float(request.form['yearbuilt'])
        bedrooms = float(request.form['bedrooms'])
        fullbath = float(request.form['fullbath'])
        garagecars = float(request.form['garagecars'])

        # ML Model Input
        features = pd.DataFrame([{
            'LotArea': lotarea,
            'OverallQual': overallqual,
            'YearBuilt': yearbuilt,
            'BedroomAbvGr': bedrooms,
            'FullBath': fullbath,
            'GarageCars': garagecars
        }])

        prediction = model.predict(features)

        base_price = max(0, prediction[0])

        # ---------------------------
        # CITY MULTIPLIERS
        # ---------------------------

        city_multiplier = {
            "Mumbai": 1.35,
            "Delhi": 1.25,
            "Bangalore": 1.20,
            "Hyderabad": 1.15,
            "Chennai": 1.10,
            "Pune": 1.08,
            "Kolkata": 1.05,
            "Ahmedabad": 1.00,
            "Jaipur": 0.95,
            "Visakhapatnam": 0.92,
            "Vijayawada": 0.90,
            "Tirupati": 0.88
        }

        base_price *= city_multiplier.get(city, 1.0)

        # ---------------------------
        # PROPERTY TYPE MULTIPLIERS
        # ---------------------------

        property_multiplier = {
            "Apartment": 1.00,
            "Villa": 1.35,
            "Independent House": 1.20,
            "Penthouse": 1.50
        }

        base_price *= property_multiplier.get(
            property_type,
            1.0
        )

        # ---------------------------
        # PRICE RANGE
        # ---------------------------

        lower_price = round(base_price * 0.95)
        upper_price = round(base_price * 1.05)

        # ---------------------------
        # PRICE PER SQFT
        # ---------------------------

        if lotarea > 0:
            price_per_sqft = round(base_price / lotarea)
        else:
            price_per_sqft = 0

        # ---------------------------
        # PROPERTY QUALITY
        # ---------------------------

        if overallqual >= 8:
            summary = "Premium Luxury Property ⭐⭐⭐⭐⭐"
            investment = "Excellent Investment"
        elif overallqual >= 6:
            summary = "Good Family Property 🏡"
            investment = "Good Investment"
        else:
            summary = "Basic Residential Property"
            investment = "Average Investment"

        # ---------------------------
        # MARKET TREND
        # ---------------------------

        growth_cities = [
            "Bangalore",
            "Hyderabad",
            "Mumbai",
            "Pune"
        ]

        if city in growth_cities:
            market_trend = "📈 High Growth Market"
        else:
            market_trend = "📊 Stable Market"

        # ---------------------------
        # RESULT TEXT
        # ---------------------------

        prediction_text = f"""
🏠 Estimated Market Value

₹{lower_price:,.0f} - ₹{upper_price:,.0f}

📍 Location: {city}

🏡 Property Type: {property_type}

📐 Price / Sq.ft: ₹{price_per_sqft:,}

⭐ Property Grade:
{summary}

💹 Investment Rating:
{investment}

📊 Market Trend:
{market_trend}
"""

        return render_template(
            'index.html',
            prediction_text=prediction_text
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)