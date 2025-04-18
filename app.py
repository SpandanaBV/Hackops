import os
import joblib
import numpy as np
from flask import Flask, request, render_template, flash
from transformers import pipeline
import math

app = Flask(__name__)

# Load models and encoders
diagnosis_model = joblib.load('models/diagnosis_model.pkl')
insurance_model = joblib.load('models/insurance_model.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_fh = joblib.load('models/le_fh.pkl')
le_symptoms = joblib.load('models/le_symptoms.pkl')
le_diagnosis = joblib.load('models/le_diagnosis.pkl')
le_insurance = joblib.load('models/le_insurance.pkl')

# Load NLP model for research papers (example: BERT-based model)
research_model = pipeline("text-classification", model="your-model-here")

# Example list of doctor locations (latitude, longitude)
doctor_locations = [
    {"name": "Dr. Smith", "specialty": "Cardiology", "latitude": 12.9716, "longitude": 77.5946},
    {"name": "Dr. John", "specialty": "Neurology", "latitude": 13.0350, "longitude": 77.5974},
    {"name": "Dr. Lee", "specialty": "Orthopedic", "latitude": 13.0844, "longitude": 77.5076}
]

# Function to calculate distance between two lat/lon points using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # in kilometers
    return distance

# Function to make research paper-based predictions
def research_paper_prediction(symptoms):
    # Example research text based on symptoms
    research_text = f"Research paper insights related to {symptoms}"
    prediction = research_model(research_text)
    return prediction[0]['label']  # This returns the research category

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        symptoms = request.form['symptoms'].strip()
        age = int(request.form['age'])
        gender = request.form['gender']
        family_history = request.form['family_history']
        billing = int(request.form['billing'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # Encode input data
        symptoms_enc = le_symptoms.transform([symptoms])[0]
        gender_enc = le_gender.transform([gender])[0]
        family_history_enc = le_fh.transform([family_history])[0]

        input_data = np.array([[symptoms_enc, age, gender_enc, family_history_enc, billing]])

        # Predict diagnosis and insurance
        diagnosis_category_enc = diagnosis_model.predict(input_data)[0]
        diagnosis_category = le_diagnosis.inverse_transform([diagnosis_category_enc])[0]

        insurance_category_enc = insurance_model.predict(input_data)[0]
        insurance_category = le_insurance.inverse_transform([insurance_category_enc])[0]

        # Find nearest doctor based on geolocation
        nearest_doctor = None
        min_distance = float('inf')
        for doctor in doctor_locations:
            distance = calculate_distance(latitude, longitude, doctor["latitude"], doctor["longitude"])
            if distance < min_distance:
                min_distance = distance
                nearest_doctor = doctor["name"]

        # Get research paper-based prediction
        research_insight = research_paper_prediction(symptoms)

        return render_template('result.html', diagnosis=diagnosis_category, insurance=insurance_category, doctor=nearest_doctor, research_insight=research_insight)

    except Exception as e:
        flash(f"An error occurred: {e}", 'error')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
