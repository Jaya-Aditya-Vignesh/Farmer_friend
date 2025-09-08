from flask import Flask, request, jsonify

app = Flask(__name__)

# This data can be expanded or moved to its own database later.
CROP_DATA = {
    "rice": [
        {"stage": "Land Preparation", "date": "May 20 - June 5"},
        {"stage": "Planting/Sowing", "date": "June 10 - June 25"},
        {"stage": "Fertilizer (1st dose)", "date": "July 10"},
        {"stage": "Harvesting", "date": "October 15 - November 10"},
    ],
    "wheat": [
        {"stage": "Sowing", "date": "November 1 - November 20"},
        {"stage": "First Irrigation", "date": "21 days after sowing"},
        {"stage": "Harvesting", "date": "March 25 - April 15"},
    ],
    "tomato": [
        {"stage": "Nursery Sowing", "date": "June - July for autumn crop"},
        {"stage": "Transplanting", "date": "25-30 days after sowing"},
        {"stage": "First Fertilizing", "date": "20-25 days after transplanting"},
        {"stage": "Harvesting", "date": "Starts 70 days after transplanting"},
    ]
}


@app.route("/api/crop-calendar", methods=['GET'])
def get_crop_calendar():
    crop_name = request.args.get('crop', '').lower()

    if not crop_name:
        return jsonify({"error": "Crop parameter is required"}), 400

    calendar = CROP_DATA.get(crop_name)

    if calendar:
        return jsonify(calendar)
    else:
        return jsonify({"error": "Calendar for the specified crop not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5003)