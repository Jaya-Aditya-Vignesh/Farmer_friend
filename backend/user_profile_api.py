import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
# --- Firebase Initialization ---
# Place the downloaded serviceAccountKey.json in your backend directory

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


app = Flask(__name__)
CORS(app)



@app.route("/api/profile/<string:user_id>", methods=['GET'])
def get_user_profile(user_id):
    """Fetches a user's profile from Firestore."""
    try:
        # Get the main user document
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()

        # Get active location from the sub-collection
        locations_ref = user_ref.collection('locations').where('active', '==', True).limit(1)
        locations = list(locations_ref.stream())
        active_location = locations[0].to_dict() if locations else None

        # Get active crop from the sub-collection
        crops_ref = user_ref.collection('crops').where('active', '==', True).limit(1)
        crops = list(crops_ref.stream())
        active_crop = crops[0].to_dict()['name'] if crops else None

        response = {
            "userId": user_id,
            "language": user_data.get('language', 'en'),
            "location": active_location,
            "chosenCrop": active_crop
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/profile/<string:user_id>", methods=['POST'])
def create_or_update_profile(user_id):
    """Creates or updates a user's profile in Firestore."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        user_ref = db.collection('users').document(user_id)

        # Set user language (creates the user document if it doesn't exist)
        user_ref.set({
            'language': data.get('language', 'en'),
            'createdAt': firestore.SERVER_TIMESTAMP
        }, merge=True)

        # Update location in the sub-collection
        if 'location' in data and data['location']:
            loc = data['location']
            # A transaction can be used here to ensure data consistency
            locations_ref = user_ref.collection('locations')
            # Deactivate old locations (optional, or just add new ones)
            # Add new active location
            locations_ref.add({
                'name': loc['name'],
                'lat': loc['lat'],
                'lng': loc['lng'],
                'active': True
            })

        # Update crop in the sub-collection
        if 'chosenCrop' in data and data['chosenCrop']:
            crops_ref = user_ref.collection('crops')
            crops_ref.add({
                'name': data['chosenCrop'],
                'active': True
            })

        return jsonify({"message": f"Profile for user {user_id} updated successfully."}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5005)