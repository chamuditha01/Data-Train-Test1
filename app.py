from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
from tensorflow.keras.models import load_model

# Correct file path format
model = load_model(r'C:\codes2\Data Train Test 1\plant_disease_classifier.keras')


# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            img_array = prepare_image(file_path)
            prediction = model.predict(img_array)
            result = 'Healthy' if prediction[0][0] > 0.5 else 'Diseased'

            return render_template('index.html', result=result, image_path=file_path)

    return render_template('index.html', result=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
