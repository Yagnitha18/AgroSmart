import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
import psycopg2

app = Flask(__name__)

# Database configuration
db_config = {
    'dbname': 'crop_recommendation_db',
    'user': 'postgres',  # Your PostgreSQL username
    'password': 'yagnitha',  # Your PostgreSQL password
    'host': 'localhost',
    'port': '5432'  # Default PostgreSQL port
}

# Function to get a database connection
def get_db_connection():
    conn = psycopg2.connect(**db_config)
    return conn

# Load dataset
data = pd.read_csv('crop_recommendation.csv')

# Feature and label selection
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features
y = data['label']  # Target (crops)

# Encode the target labels (crops)
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Model training using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model for later use
pickle.dump(model, open('crop_model.pkl', 'wb'))

# Load the model
model = pickle.load(open('crop_model.pkl', 'rb'))

# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    crop_index = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])[0]
    crop_name = le.inverse_transform([crop_index])[0]
    return crop_name

@app.route('/')
def home():
    return render_template('app4.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    # Recommend the crop
    recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

    # Insert the data into the database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO crops (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop_label) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (N, P, K, temperature, humidity, ph, rainfall, recommended_crop)
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'recommended_crop': recommended_crop})

@app.route('/download', methods=['GET'])
def download():
    # Specify the directory where your exported file is located
    export_directory = os.getcwd()  # Change if necessary
    return send_from_directory(directory=export_directory, path='exported_crops.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
