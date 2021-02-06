from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
import json
from PIL import Image
from base64 import decodestring
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import imghdr
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route("/")
def index():
    print(request)
    return render_template('test.html', pred=0)

@app.route('/predict', methods=['POST'])
def predict():
    image_size = 299
    print(request)
    print(request.data)
    nparr = np.frombuffer(request.data, np.uint8)
    print(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
    cv2.imwrite('test.jpg',img)

    return str(img.shape)

@app.route('/', methods=['POST'])
def upload_file():
    print("Update - ", end='')
    print(request.files)
    uploaded_file = request.files['image_uploads']
    filename = secure_filename(uploaded_file.filename)
    print(filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))


def main():
    """Run the app."""
    app.run(host='0.0.0.0', debug=False)  # nosec


if __name__ == '__main__':
    main()
