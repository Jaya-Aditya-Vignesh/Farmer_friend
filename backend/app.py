import os
import io
import time
import pickle
import requests
import joblib
import firebase_admin
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from geopy.geocoders import Nominatim
from firebase_admin import credentials, firestore

# --- INITIALIZATION ---
load_dotenv()
app = Flask(__name__)

# Configure CORS to allow requests from your Next.js frontend
CORS(app, origins=["http://localhost:3000"], allow_headers=["Content-Type"])

# --- Firebase Initialization ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    db = None

# --- ML Model Loading ---
CROP_MODEL = None
MAIN_SCALER = None
PHOSPHORUS_MODEL = None
POTASSIUM_MODEL = None
PEST_MODEL = None
PEST_CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___healthy',
                    'Grape___healthy', 'Potato___healthy', 'Tomato___healthy']

try:
    CROP_MODEL = pickle.load(open('model.pkl', 'rb'))
    print("Crop model loaded successfully")
except Exception as e:
    print(f"Warning: Crop model not loaded: {e}")

try:
    MAIN_SCALER = pickle.load(open('scaler.pkl', 'rb'))
    print("Main scaler loaded successfully")
except Exception as e:
    print(f"Warning: Main scaler not loaded: {e}")

try:
    PHOSPHORUS_MODEL = pickle.load(open('Phosphorus_model.pkl', 'rb'))
    print("Phosphorus model loaded successfully")
except Exception as e:
    print(f"Warning: Phosphorus model not loaded: {e}")

try:
    POTASSIUM_MODEL = pickle.load(open('Potassium_model.pkl', 'rb'))
    print("Potassium model loaded successfully")
except Exception as e:
    print(f"Warning: Potassium model not loaded: {e}")

try:
    PEST_MODEL = tf.keras.models.load_model('ml_models/pest_detection_model.h5')
    print("Pest detection model loaded successfully")
except Exception as e:
    print(f"Warning: Pest detection model not loaded: {e}")

# --- API Client Initialization ---
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Warning: Groq client not initialized: {e}")
    groq_client = None

geolocator = Nominatim(user_agent="farmer_companion_app_v3", timeout=10)


# --- HELPER FUNCTIONS ---
def get_soil_data(lat, lon):
    """Get soil data from SoilGrids API"""
    try:
        url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}&properties=ocd,phh2o&depths=0-5cm"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        properties = response.json()['properties']
        organic_carbon = properties['ocd']['mean']
        ph = properties['phh2o']['mean'] / 10.0
        return organic_carbon, ph
    except Exception as e:
        print(f"Error fetching soil data: {e}")
        # Return default values if API fails
        return 15.0, 6.5


def get_weather_data(lat, lon):
    """Get weather data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,precipitation_sum&timezone=auto"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        daily_data = response.json()['daily']
        avg_temp = sum(daily_data['temperature_2m_mean']) / len(daily_data['temperature_2m_mean'])
        total_rainfall = sum(daily_data['precipitation_sum'])
        return avg_temp, total_rainfall
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        # Return default values if API fails
        return 25.0, 100.0


def preprocess_pest_image(image_bytes, target_size=(224, 224)):
    """Preprocess image for pest detection model"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


# --- API ROUTES ---

# Health check endpoint
@app.route("/api/health", methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Backend is running"}), 200


# 1. User Profile API
@app.route("/api/profile/<string:user_id>", methods=['GET', 'POST'])
def handle_profile(user_id):
    if not db:
        return jsonify({"error": "Database not available"}), 500

    if request.method == 'GET':
        try:
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
            if not user_doc.exists:
                return jsonify({"error": "User not found"}), 404

            user_data = user_doc.to_dict()

            # Get latest location
            locations_ref = user_ref.collection('locations').order_by('createdAt',
                                                                      direction=firestore.Query.DESCENDING).limit(1)
            locations = list(locations_ref.stream())
            active_location = locations[0].to_dict() if locations else None

            # Get latest crop
            crops_ref = user_ref.collection('crops').order_by('createdAt', direction=firestore.Query.DESCENDING).limit(
                1)
            crops = list(crops_ref.stream())
            active_crop = crops[0].to_dict().get('name') if crops else None

            response = {
                "userId": user_id,
                "language": user_data.get('language', 'en'),
                "location": active_location,
                "chosenCrop": active_crop
            }
            return jsonify(response)
        except Exception as e:
            print(f"Error in GET profile: {e}")
            return jsonify({"error": f"An error occurred in GET profile: {str(e)}"}), 500

    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        try:
            user_ref = db.collection('users').document(user_id)

            @firestore.transactional
            def update_in_transaction(transaction, user_ref, update_data):
                user_update_payload = {'updatedAt': firestore.SERVER_TIMESTAMP}
                if 'language' in update_data:
                    user_update_payload['language'] = update_data['language']
                transaction.set(user_ref, user_update_payload, merge=True)

                if 'location' in update_data and update_data['location']:
                    location_ref = user_ref.collection('locations').document()
                    transaction.set(location_ref, {
                        **update_data['location'],
                        'createdAt': firestore.SERVER_TIMESTAMP
                    })

                if 'chosenCrop' in update_data and update_data['chosenCrop']:
                    crop_ref = user_ref.collection('crops').document()
                    transaction.set(crop_ref, {
                        'name': update_data['chosenCrop'],
                        'createdAt': firestore.SERVER_TIMESTAMP
                    })

            transaction = db.transaction()
            update_in_transaction(transaction, user_ref, data)

            return jsonify({"message": f"Profile for user {user_id} updated successfully."}), 200
        except Exception as e:
            print(f"Error in POST profile: {e}")
            return jsonify({"error": f"An error occurred in POST profile: {str(e)}"}), 500


# 2. Maps API
@app.route("/api/reverse-geocode", methods=['GET'])
def reverse_geocode_location():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    try:
        coordinates = f"{lat}, {lon}"
        time.sleep(1)  # Rate limiting
        location_data = geolocator.reverse(coordinates, language='en')

        if location_data:
            return jsonify({"name": location_data.address})
        else:
            return jsonify({"name": f"Lat: {float(lat):.4f}, Lng: {float(lon):.4f}"})
    except Exception as e:
        print(f"Error in reverse-geocode: {e}")
        return jsonify({"error": f"An error occurred in reverse-geocode: {str(e)}"}), 500


# 3. Crop Recommendation API
@app.route("/api/recommend-crop", methods=['GET'])
def recommend_crop():
    if not all([CROP_MODEL, MAIN_SCALER, PHOSPHORUS_MODEL, POTASSIUM_MODEL]):
        return jsonify({"error": "Required models are not loaded"}), 500

    try:
        lat_str = request.args.get('lat')
        lon_str = request.args.get('lon')

        if not lat_str or not lon_str:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        lat = float(lat_str)
        lon = float(lon_str)

        print(f"Getting recommendations for coordinates: {lat}, {lon}")

        # Get soil and weather data
        organic_carbon_as_N, ph = get_soil_data(lat, lon)
        temperature, rainfall = get_weather_data(lat, lon)

        print(f"Soil data - Organic carbon: {organic_carbon_as_N}, pH: {ph}")
        print(f"Weather data - Temperature: {temperature}, Rainfall: {rainfall}")

        # Predict phosphorus and potassium values
        p_features = [[organic_carbon_as_N, temperature, 80, ph, rainfall]]  # 80 is default humidity
        k_features = [[organic_carbon_as_N, temperature, 80, ph, rainfall]]

        p_value = PHOSPHORUS_MODEL.predict(p_features)[0]
        k_value = POTASSIUM_MODEL.predict(k_features)[0]

        print(f"Predicted - Phosphorus: {p_value}, Potassium: {k_value}")

        # Final crop prediction
        final_features = [[organic_carbon_as_N, p_value, k_value, temperature, 80, ph, rainfall]]
        scaled_final_features = MAIN_SCALER.transform(final_features)
        crop_prediction = CROP_MODEL.predict(scaled_final_features)

        print(f"Recommended crop: {crop_prediction[0]}")

        return jsonify({"recommended_crop": crop_prediction[0]})

    except ValueError as e:
        print(f"ValueError in recommend-crop: {e}")
        return jsonify({"error": "Invalid latitude or longitude format"}), 400
    except Exception as e:
        print(f"Error in recommend-crop: {e}")
        return jsonify({"error": f"An error occurred in recommend-crop: {str(e)}"}), 500


# 4. Voice Chatbot API
@app.route("/api/voice-chat", methods=['POST'])
def voice_chat():
    if not groq_client:
        return jsonify({"error": "AI service not available"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        system_prompt = "You are an AI Farmer Assistant. Provide concise, clear advice suitable for voice output."
        chat_completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        ai_response_text = chat_completion.choices[0].message.content

        tts = gTTS(text=ai_response_text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        return send_file(audio_fp, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error in voice-chat: {e}")
        return jsonify({"error": f"An error occurred in voice-chat: {str(e)}"}), 500


# 5. Pest Detection API
@app.route("/api/predict", methods=['POST'])
def predict_pest():
    if not PEST_MODEL:
        return jsonify({"error": "Pest detection model not available"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_pest_image(image_bytes)
        prediction = PEST_MODEL.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        predicted_class_name = PEST_CLASS_NAMES[predicted_class_index].replace('_', ' ').replace('___', ' - ')

        return jsonify({
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"CORS enabled for: http://localhost:3000")
    print(f"Available endpoints:")
    print(f"  GET  /api/health")
    print(f"  GET  /api/profile/<user_id>")
    print(f"  POST /api/profile/<user_id>")
    print(f"  GET  /api/reverse-geocode")
    print(f"  GET  /api/recommend-crop")
    print(f"  POST /api/voice-chat")
    print(f"  POST /api/predict")

    # Run the consolidated app on port 5000
    app.run(debug=True, port=5000, host='127.0.0.1')