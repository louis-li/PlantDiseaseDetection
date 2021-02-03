import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
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
app.config['IMAGE_SIZE'] = 299
app.config['CLASS'] = ''
def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.before_first_request
def load_model_to_app():
    app.predictor = load_model('../models/InceptionResNetV2')
    
@app.route("/")
def index():
    #pred = app.config['CLASS'] 
    return render_template('upload.html', pred='')

@app.route('/predict', methods=['POST'])
def predict():
    image_size = app.config['IMAGE_SIZE']
    print(request)
    nparr = np.frombuffer(request.data, np.uint8)
    print(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
    cv2.imwrite('test.jpg',img)
    predictions = app.predictor.predict(img.reshape(-1,image_size,image_size,3))
    print('INFO Predictions: {}'.format(predictions))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
    classnames = ['healthy','multiple diseases', 'rust', 'scab']
    return classnames[class_]

@app.route('/', methods=['POST'])
def upload_file():
    print("Upload - ", end='')
    image_size = app.config['IMAGE_SIZE']
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    print(filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        full_filename = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(full_filename)
        img = cv2.imread(full_filename, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
        #cv2.imwrite('test.jpg',img)
        predictions = app.predictor.predict(img.reshape(-1,image_size,image_size,3))
        print('INFO Predictions: {}'.format(predictions))

        class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
        classnames = ['healthy','multiple diseases', 'rust', 'scab']
        #app.config['CLASS'] = classnames[class_]
    return render_template('upload.html', pred=classnames[class_])


def main():
    """Run the app."""
    app.run(host='0.0.0.0', debug=False)  # nosec


if __name__ == '__main__':
    main()