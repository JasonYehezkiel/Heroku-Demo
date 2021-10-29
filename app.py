import os
import numpy as np
import json
from keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory
from numpy.core.numeric import full
from app_utils import *

### CONSTANT ###
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = STATIC_FOLDER + '/tmp'
### CONSTANT ###

with open('static/params/params4.json', 'r') as json_file:
    data = json.load(json_file)

app = Flask(__name__)


def predict(fullpath):
    # parameter dari model kita
    params = {k: np.array(v) for k, v in data.items()}
    # pra-pemrosesan gambar
    img = image.load_img(fullpath, target_size=(64, 64, 3))
    input_arr = image.img_to_array(img)
    input_arr = np.array([input_arr])

    img2 = np.expand_dims(input_arr, axis=0)
    img2 = img2.reshape(img2.shape[0], -1).T
    img2 = img2.astype('float') / 255
    al, _ = L_model_forward(img2, params)

    return al


# Homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)

        pred_prob = result

        if pred_prob > .5:
            label = 'With_mask'
            accuracy = np.round(pred_prob * 100, 2)
        else:
            label = 'Without_mask'
            accuracy = np.round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=True)
