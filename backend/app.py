import os
import io
import time
import pickle
import requests
import joblib
import firebase_admin
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import json
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
CORS(app, origins=["http://localhost:3000"])

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
PEST_MODEL_PYTORCH = None  # PyTorch model
PYTORCH_DEVICE = None
CLASS_TO_IDX = None
IDX_TO_CLASS = None
PYTORCH_TRANSFORM = None

# Load traditional ML models
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

# Load PyTorch pest detection model from the new .pkl file
try:
    # Set device
    PYTORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {PYTORCH_DEVICE}")

    # Define file paths based on your provided structure
    model_path = os.path.join('ml_models', 'pest_model.pkl')
    labels_path = 'labels.json'

    # Load class mapping from the new label_map.json
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            CLASS_TO_IDX = json.load(f)
            IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
            print(f"Loaded {len(CLASS_TO_IDX)} classes: {list(CLASS_TO_IDX.keys())}")
    else:
        print(f"Warning: {labels_path} not found.")

    # Load the new .pkl model
    if os.path.exists(model_path) and CLASS_TO_IDX:
        # Load model using torch.load as shown in your test.py file
        PEST_MODEL_PYTORCH = torch.load(model_path, map_location=PYTORCH_DEVICE)
        PEST_MODEL_PYTORCH.eval()
        PEST_MODEL_PYTORCH.to(PYTORCH_DEVICE)

        # Define preprocessing transform based on your test.py file
        PYTORCH_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print("PyTorch pest detection model loaded successfully")
    else:
        print(f"Warning: PyTorch model checkpoint not found at {model_path} or labels.json is missing.")

except Exception as e:
    print(f"Warning: PyTorch pest detection model not loaded: {e}")
    PEST_MODEL_PYTORCH = None

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
        return 25.0, 100.0


def preprocess_pest_image_pytorch(image_bytes):
    """Preprocess image for PyTorch pest detection model"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = PYTORCH_TRANSFORM(img).unsqueeze(0).to(PYTORCH_DEVICE)
        return input_tensor
    except Exception as e:
        print(f"Error preprocessing image for PyTorch: {e}")
        raise


def predict_with_pytorch_model(image_bytes):
    """Predict using PyTorch model"""
    if not PEST_MODEL_PYTORCH or not PYTORCH_TRANSFORM or not IDX_TO_CLASS:
        raise Exception("PyTorch model not available")

    input_tensor = preprocess_pest_image_pytorch(image_bytes)

    with torch.no_grad():
        outputs = PEST_MODEL_PYTORCH(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    predicted_class = IDX_TO_CLASS[pred_idx.item()]
    confidence = conf.item() * 100

    return predicted_class, confidence


def get_enhanced_pest_advice(prediction):
    """Get enhanced advice based on prediction"""
    advice_map = {
        # General advice for common issues from the new labels
        'Apple_Apple_scab': {
            'treatment': 'Apply fungicide treatments (Captan, Myclobutanil) every 10-14 days during wet periods. Prune to improve air circulation.',
            'prevention': 'Remove fallen leaves, ensure good air circulation, and use resistant varieties.',
            'severity': 'moderate'
        },
        'Apple_Black_rot': {
            'treatment': 'Remove infected fruits and branches immediately. Apply copper-based fungicides during dormant season.',
            'prevention': 'Prune for air circulation, avoid overhead watering, and remove mummified fruits promptly.',
            'severity': 'high'
        },
        'Apple_healthy': {
            'treatment': 'Continue current care practices.',
            'prevention': 'Regular monitoring, balanced fertilization, and proper pruning.',
            'severity': 'none'
        },
        'Grape_Black_rot': {
            'treatment': 'Remove and destroy infected berries. Apply fungicides before and after flowering.',
            'prevention': 'Maintain good air circulation and use resistant grape varieties.',
            'severity': 'high'
        },
        'Potato_Late_blight': {
            'treatment': 'Apply protective fungicides. Remove and destroy infected plant parts immediately.',
            'prevention': 'Use certified seed potatoes and ensure good plant spacing.',
            'severity': 'high'
        },
        'Tomato_Tomato_Yellow_Leaf_Curl_Virus': {
            'treatment': 'There is no chemical cure. Remove and destroy infected plants immediately.',
            'prevention': 'Control whiteflies, which are the vector. Use reflective mulch and insect-proof nets.',
            'severity': 'critical'
        },
        'Tomato_healthy': {
            'treatment': 'Maintain current growing conditions.',
            'prevention': 'Monitor for pests and diseases like blight and wilt.',
            'severity': 'none'
        },
        # Add more specific advice here for other labels as needed
    }

    return advice_map.get(prediction, {
        'treatment': 'Consult local agricultural experts for specific treatment recommendations.',
        'prevention': 'Regular monitoring and good cultural practices are key.',
        'severity': 'unknown'
    })


def get_concise_prompt(language):
    """Get system prompt based on language"""
    prompts = {
        'en': "You are an AI Farmer Assistant. Provide extremely concise, clear farming advice in 1-2 sentences maximum. Be direct and practical.",
        'hi': "आप एक AI कृषक सहायक हैं। अधिकतम 1-2 वाक्यों में अत्यंत संक्षिप्त, स्पष्ट कृषि सलाह दें। प्रत्यक्ष और व्यावहारिक बनें।",
        'ta': "நீங்கள் ஒரு AI விவசாய உதவியாளர். அதிகபட்சம் 1-2 வாக்கியங்களில் மிகவும் சுருக்கமான, தெளிவான விவசாய ஆலோசனை வழங்கவும். நேரடியாகவும் நடைமுறையாகவும் இருங்கள்.",
        'te': "మీరు AI వ్యవసాయ సహాయకుడు. గరిష్టంగా 1-2 వాక్యాలలో అత్యంత సంక్షిప్తమైన, స్పష్టమైన వ్యవసాయ సలహా ఇవ్వండి. ప్రత్యక్షంగా మరియు ఆచరణాత్మకంగా ఉండండి.",
        'kn': "ನೀವು AI ಕೃಷಿ ಸಹಾಯಕ. ಗರಿಷ್ಠ 1-2 ವಾಕ್ಯಗಳಲ್ಲಿ ಅತ್ಯಂತ ಸಂಕ್ಷಿಪ್ತ, ಸ್ಪಷ್ಟ ಕೃಷಿ ಸಲಹೆ ನೀಡಿ. ನೇರ ಮತ್ತು ಪ್ರಾಯೋಗಿಕವಾಗಿರಿ.",
        'ml': "നിങ്ങൾ ഒരു AI കാർഷിക സഹായകനാണ്. പരമാവധി 1-2 വാക്യങ്ങളിൽ അതീവ സംക്ഷിപ്തവും വ്യക്തവുമായ കാർഷിക ഉപദേശം നൽകുക. നേരിട്ടും പ്രായോഗികവുമാകുക.",
        'bn': "আপনি একজন AI কৃষক সহায়ক। সর্বোচ্চ ১-২ বাক্যে অত্যন্ত সংক্ষিপ্ত, স্পষ্ট কৃষি পরামর্শ দিন। প্রত্যক্ষ এবং ব্যবহারিক হন।",
        'gu': "તમે AI કૃષક સહાયક છો. મહત્તમ 1-2 વાક્યોમાં અત્યંત સંક્ષિપ્ત, સ્પષ્ટ કૃષિ સલાહ આપો. પ્રત્યક્ષ અને વ્યવહારિક બનો.",
        'mr': "तुम्ही AI शेतकरी सहाyyक आहात. कमाल 1-2 वाक्यांत अत्यंत संक्षिप्त, स्पष्ट शेती सल्ला द्या. थेट आणि व्यावहारिक व्हा.",
        'pa': "ਤੁਸੀਂ AI ਕਿਸਾਨ ਸਹਾਇਕ ਹੋ। ਵੱਧ ਤੋਂ ਵੱਧ 1-2 ਵਾਕਾਂ ਵਿੱਚ ਬਹੁਤ ਸੰਖੇਪ, ਸਪੱਸ਼ਟ ਖੇਤੀ ਸਲਾਹ ਦਿਓ। ਸਿੱਧੇ ਅਤੇ ਵਿਹਾਰਕ ਬਣੋ।"
    }
    return prompts.get(language, prompts['en'])


# --- DATABASE FUNCTIONS ---
def get_pesticide_details_from_db(query):

    if not db:
        return None
    try:
        pesticides_ref = db.collection('pesticides')

        query_words = query.lower().split()

        for word in query_words:
            results = pesticides_ref.where('targetPests', 'array_contains', word).stream()

            for doc in results:
                return doc.to_dict()

        return None
    except Exception as e:
        print(f"Error getting pesticide from DB: {e}")
        return None


def populate_pesticides_db():
    """
    Populates the 'pesticides' collection in Firebase with sample data.
    UNCOMMENT AND RUN THIS FUNCTION ONCE to set up your database.
    """
    if not db:
        print("Database not available. Cannot populate.")
        return

    pesticides_ref = db.collection('pesticides')

    sample_pesticides = [
        {
            "name": "Bayer Fungicide",
            "activeIngredients": "Propiconazole",
            "description": "A broad-spectrum fungicide effective against a variety of fungal diseases.",
            "applicationInstructions": "Mix 2ml per liter of water. Apply every 7-14 days. Avoid applying during rain.",
            "targetPests": ["apple_scab", "black_rot", "powdery_mildew"]
        },
        {
            "name": "Neem Oil",
            "activeIngredients": "Azadirachtin",
            "description": "A natural insecticide, fungicide, and miticide.",
            "applicationInstructions": "Mix 1-2 tablespoons per gallon of water with a mild soap. Spray on all plant surfaces.",
            "targetPests": ["aphids", "spider_mites", "whiteflies", "thrips"]
        },
        {
            "name": "Mancozeb",
            "activeIngredients": "Mancozeb",
            "description": "Protective fungicide for control of early and late blight.",
            "applicationInstructions": "Apply preventatively at the onset of favorable disease conditions.",
            "targetPests": ["early_blight", "late_blight", "septoria_leaf_spot"]
        }
    ]

    try:
        for pesticide in sample_pesticides:
            pesticides_ref.add(pesticide)
        print("Pesticides collection populated successfully.")
    except Exception as e:
        print(f"Failed to populate pesticides DB: {e}")


# --- API ROUTES ---

# Health check endpoint
@app.route("/api/health", methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "message": "Backend is running",
        "models": {
            "crop_model": CROP_MODEL is not None,
            "pest_detection_pytorch": PEST_MODEL_PYTORCH is not None,
            "pytorch_classes": len(CLASS_TO_IDX) if CLASS_TO_IDX else 0
        }
    }
    return jsonify(status), 200


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

# 6. Crop Calendar API
@app.route("/api/crop-calendar", methods=['POST'])
def crop_calendar():
    if not groq_client:
        return jsonify({"error": "AI service not available"}), 500

    data = request.get_json()
    if not data or 'crop' not in data or 'lat' not in data or 'lon' not in data:
        return jsonify({"error": "Invalid JSON, 'crop', 'lat', and 'lon' are required"}), 400

    crop = data['crop']
    lat = data['lat']
    lon = data['lon']

    try:
        # Get weather data for the location
        temperature, rainfall = get_weather_data(float(lat), float(lon))

        # Compose a concise prompt for Groq
        prompt = (
            f"You are an AI Farmer Assistant. Generate a concise crop calendar for {crop} "
            f"at latitude {lat}, longitude {lon}. The average temperature is {temperature:.1f}°C "
            f"and total rainfall is {rainfall:.1f}mm. List key events (sowing, fertilizing, irrigation, harvesting) "
            f"with recommended short date ranges only. Just return key events, Nothing else. Limit to 5 events."
        )

        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )
        calendar_text = chat_completion.choices[0].message.content.strip()

        return jsonify({
            "crop": crop,
            "calendar": calendar_text,
            "location": {"lat": lat, "lon": lon},
            "temperature": temperature,
            "rainfall": rainfall
        })
    except Exception as e:
        print(f"Error in crop-calendar: {e}")
        return jsonify({"error": f"An error occurred in crop-calendar: {str(e)}"}), 500
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
        hum=np.random.randint(14,90)
        # Predict phosphorus and potassium values
        p_features = [[organic_carbon_as_N, temperature, hum, ph, rainfall]]  # 80 is default humidity
        k_features = [[organic_carbon_as_N, temperature,hum, ph, rainfall]]

        p_value = PHOSPHORUS_MODEL.predict(p_features)[0]
        k_value = POTASSIUM_MODEL.predict(k_features)[0]

        print(f"Predicted - Phosphorus: {p_value}, Potassium: {k_value}, humidity: {hum}")

        # Final crop prediction
        final_features = [[organic_carbon_as_N, p_value, k_value, temperature, hum, ph, rainfall]]
        #scaled_final_features = MAIN_SCALER.transform(final_features)
        crop_prediction = CROP_MODEL.predict(final_features)

        print(f"Recommended crop: {crop_prediction[0]}")

        return jsonify({"recommended_crop": crop_prediction[0]})

    except ValueError as e:
        print(f"ValueError in recommend-crop: {e}")
        return jsonify({"error": "Invalid latitude or longitude format"}), 400
    except Exception as e:
        print(f"Error in recommend-crop: {e}")
        return jsonify({"error": f"An error occurred in recommend-crop: {str(e)}"}), 500


# 4. Enhanced Voice Chatbot API
@app.route("/api/voice-chat", methods=['POST'])
def voice_chat():
    if not groq_client:
        return jsonify({"error": "AI service not available"}), 500
    if not db:
        return jsonify({"error": "Database not available"}), 500

    data = request.get_json()
    if not data or 'message' not in data or 'userId' not in data:
        return jsonify({"error": "Invalid JSON, 'message' and 'userId' are required"}), 400

    user_message = data.get('message')
    user_id = data.get('userId')
    voice_input = data.get('voiceInput', False)
    request_audio = data.get('requestAudio', voice_input)
    language = 'en'

    # Check if the user's message contains a pesticide-related keyword
    pesticide_query_words = ['pesticide', 'treatment', 'cure', 'spray', 'control']
    found_pesticide_query = any(word in user_message.lower() for word in pesticide_query_words)
    pesticide_data = None

    if found_pesticide_query:
        # Extract the pest name (a simple approach)
        pest_name = user_message.lower().replace("what is the", "").replace("pesticide for", "").strip()
        pesticide_data = get_pesticide_details_from_db(pest_name)

    try:
        # Get user's language from Firestore
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            language = user_data.get('language', 'en')

        # Generate a dynamic system prompt for Groq
        system_prompt = get_concise_prompt(language)
        if pesticide_data:
            # Augment the prompt with specific pesticide details
            system_prompt += (
                f"\n\nHere are relevant details about a pesticide to answer the user's query: "
                f"Name: {pesticide_data.get('name')}, "
                f"Active Ingredients: {pesticide_data.get('activeIngredients')}, "
                f"Description: {pesticide_data.get('description')}, "
                f"Application: {pesticide_data.get('applicationInstructions')}. "
                f"Use this information to provide a helpful response and a recommendation."
            )

        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=250,  # Increased for pesticide recommendations
            temperature=0.2
        )
        ai_response_text = chat_completion.choices[0].message.content

        # Clean up response
        ai_response_text = ai_response_text.strip()
        if len(ai_response_text) > 400:
            ai_response_text = ai_response_text[:397] + "..."

        print(f"AI Response: {ai_response_text}")
        print(f"Language: {language}, Voice input: {voice_input}, Request audio: {request_audio}")

        # If audio is requested
        if request_audio:
            try:
                tts = gTTS(text=ai_response_text, lang=language, slow=False)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)

                return send_file(
                    audio_fp,
                    mimetype="audio/mpeg",
                    as_attachment=False,
                    download_name=f"response_{int(time.time())}.mp3"
                )
            except Exception as tts_error:
                print(f"TTS Error: {tts_error}")
                return jsonify({
                    "text": ai_response_text,
                    "audio_error": "Audio generation failed, text response provided"
                })
        else:
            return jsonify({
                "text": ai_response_text,
                "language": language,
                "context": "pest_aware" if pesticide_data else "general"
            })

    except Exception as e:
        print(f"Error in voice-chat: {e}")
        return jsonify({"error": f"An error occurred in voice-chat: {str(e)}"}), 500


# 5. Enhanced Pest Detection API with only the new PyTorch model
@app.route("/api/predict", methods=['POST'])
def predict_pest():
    if not PEST_MODEL_PYTORCH:
        return jsonify({"error": "Pest detection model not available"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()

        # Predict using the PyTorch model
        predicted_class_name, confidence = predict_with_pytorch_model(image_bytes)
        model_used = 'pytorch'

        # Get enhanced advice
        advice = get_enhanced_pest_advice(predicted_class_name)

        # Clean up class name for display
        display_name = predicted_class_name.replace('_', ' ').replace('___', ' - ')

        result = {
            "prediction": display_name,
            "raw_prediction": predicted_class_name,
            "confidence": f"{confidence:.2f}%",
            "confidence_score": confidence,
            "model_used": model_used,
            "advice": advice,
            "timestamp": int(time.time())
        }

        print(f"Pest detection result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# New endpoint to get available models info
@app.route("/api/models/info", methods=['GET'])
def get_models_info():
    """Get information about available models"""
    info = {
        "pest_detection": {
            "pytorch": {
                "available": PEST_MODEL_PYTORCH is not None,
                "classes": list(CLASS_TO_IDX.keys()) if CLASS_TO_IDX else []
            }
        },
        "crop_recommendation": {
            "available": CROP_MODEL is not None
        },
        "voice_chat": {
            "available": groq_client is not None
        }
    }
    return jsonify(info), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # UNCOMMENT THE LINE BELOW AND RUN ONCE TO POPULATE YOUR DATABASE
    # WITH SAMPLE PESTICIDE DATA. THEN COMMENT IT OUT AGAIN.
    # populate_pesticides_db()

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
    print(f"  GET  /api/models/info")

    # Run the consolidated app on port 5000
    app.run(debug=True, port=5000, host='127.0.0.1')