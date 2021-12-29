from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
# import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model_ann = load_model('models/daun_sirih_seledri_model_ann.h5')
model_cnn = load_model('models/daun_sirih_seledri_model_cnn.h5')


def predict(img_path, model_type):
    x = load_img(img_path, target_size=(32,32))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    if model_type == 'ann':
        array = model_ann.predict(x)
    elif model_type == 'cnn':
        array = model_cnn.predict(x)
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
            result = predict(img_path, 'ann')
        return render_template('ann.html', hasil=result)
    return render_template('ann.html')

@app.route("/cnn", methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            result = predict(img_path, 'cnn')
        return render_template('cnn.html', hasil=result)
    return render_template('cnn.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/about")
def about():
    return render_template('about.html')