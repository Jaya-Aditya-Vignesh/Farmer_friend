from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

MODEL = None
try:
    MODEL = tf.keras.models.load_model('ml_models/pest_detection_model.h5')
    CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___healthy',
                   'Grape___healthy', 'Potato___healthy', 'Tomato___healthy']  # Example class names
except Exception as e:
    print(f"Warning: Could not load model. Error: {e}")


def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/api/predict", methods=['POST'])
def predict_pest():
    if MODEL is None:
        return jsonify({"error": "Model is not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        prediction = MODEL.predict(processed_image)

        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        predicted_class_name = CLASS_NAMES[predicted_class_index].replace('_', ' ').replace('___', ' - ')

        return jsonify({
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5004)