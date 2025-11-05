from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/pomegranate_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (same order as training)
CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

# Optional: short descriptions
DISEASE_INFO = {
    "Alternaria": {
        "desc": "Alternaria fruit spot causes dark, sunken lesions on the fruit surface.",
        "solution": "Remove infected fruits and apply a copper-based fungicide."
    },
    "Anthracnose": {
        "desc": "Anthracnose causes circular sunken black spots on fruit and leaves.",
        "solution": "Use clean planting material and apply Mancozeb spray."
    },
    "Bacterial_Blight": {
        "desc": "This disease leads to brown, water-soaked lesions on leaves and fruit.",
        "solution": "Prune infected branches and spray with Streptocycline."
    },
    "Cercospora": {
        "desc": "Cercospora spot results in grey or brown lesions on fruit and leaves.",
        "solution": "Maintain field hygiene and apply Carbendazim-based fungicide."
    },
    "Healthy": {
        "desc": "The fruit appears healthy with no visible disease symptoms.",
        "solution": "Continue regular monitoring and good agricultural practices."
    }
}

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions) * 100)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    info = DISEASE_INFO[predicted_class]

    return render_template(
        'result.html',
        disease=predicted_class,
        confidence=round(confidence, 2),
        image_path=filepath,
        desc=info['desc'],
        solution=info['solution']
    )

if __name__ == '__main__':
    app.run(debug=True)
