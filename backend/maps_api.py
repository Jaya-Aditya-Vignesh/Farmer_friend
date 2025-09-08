from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim

app = Flask(__name__)

geolocator = Nominatim(user_agent="farmer_companion_app_v1")

@app.route("/api/geocode", methods=['GET'])
def geocode_location():
    location_name = request.args.get('location')

    if not location_name:
        return jsonify({"error": "Location parameter is required"}), 400

    try:
        location_data = geolocator.geocode(location_name)
        if location_data:
            response = {
                "name": location_data.address,
                "lat": location_data.latitude,
                "lng": location_data.longitude
            }
            return jsonify(response)
        else:
            return jsonify({"error": "Location not found"}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)