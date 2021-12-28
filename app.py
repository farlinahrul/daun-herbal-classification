from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
# import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('models/daun_sirih_seledri_model_cnn.h5')

def predict(img_path):
    x = load_img(img_path, target_size=(32,32))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    # answer = np.argmax(result)
    if result<=0.5 :
        answer = "Seledri"
    else :
        answer = "Sirih"
    return answer

@app.route("/")
def hello_world():
    return render_template('template_header.html')

@app.route("/ann", methods=['GET', 'POST'])
def ann():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            result = predict(img_path)
        return render_template('ann.html', hasil=result)
    return render_template('ann.html')

@app.route("/cnn", methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        return render_template('cnn.html')
    return render_template('cnn.html')