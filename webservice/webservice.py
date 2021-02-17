import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import imghdr
import os
import json

app = Flask(__name__)
app.config['MODEL_PATH'] = '../models/EfficientNetB2'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', 'jpeg']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['IMAGE_SIZE'] = 224
app.config['CLASS'] =  ['Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
       'Apple scab', 'Bell Pepper Bacterial spot', 'Bell Pepper healthy',
       'Blueberry healthy', 'Cherry (including sour) Powdery mildew',
       'Cherry (including sour) healthy',
       'Corn (maize) Cercospora leaf spot Gray leaf spot',
       'Corn (maize) Common rust ', 'Corn (maize) Northern Leaf Blight',
       'Corn (maize) healthy', 'Grape Black rot',
       'Grape Esca (Black Measles)',
       'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy',
       'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot',
       'Peach healthy', 'Potato Early blight', 'Potato Late blight',
       'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
       'Squash Powdery mildew', 'Strawberry Leaf scorch',
       'Strawberry healthy', 'Tomato Bacterial spot',
       'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
       'Tomato Septoria leaf spot',
       'Tomato Spider mites Two-spotted spider mite',
       'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus',
       'Tomato healthy', 'Tomato mosaic virus']

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

@app.before_first_request
def load_model_to_app():
    print(app.config['MODEL_PATH'])
    app.predictor = load_model(app.config['MODEL_PATH'])
    
@app.route("/")
def index():
    #files = os.listdir(app.config['UPLOAD_PATH'])
    #print('files: ', files)
    return render_template('index.html', pred='', files = [])

@app.route("/qna")
def qna():
    #files = os.listdir(app.config['UPLOAD_PATH'])
    #print('files: ', files)
    return render_template('qna.html')

@app.route("/qna2")
def qna2():
    #files = os.listdir(app.config['UPLOAD_PATH'])
    #print('files: ', files)
    return render_template('qna2.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_size = app.config['IMAGE_SIZE']
    print(request)
    nparr = np.frombuffer(request.data, np.uint8)
    print(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
    #cdcv2.imwrite('test.jpg',img)
    predictions = app.predictor.predict(img.reshape(-1,image_size,image_size,3))
    print('INFO Predictions: {}'.format(predictions))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
    classnames = app.config['CLASS']
    return classnames[class_]

@app.route('/', methods=['POST'])
def upload_file():
    print("Upload - ", end='')
    image_size = app.config['IMAGE_SIZE']
    uploaded_file = request.files['image_uploads']
    filename = secure_filename(uploaded_file.filename).lower()
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
        classnames = app.config['CLASS']
        #files = getfiles(app.config['UPLOAD_PATH'])
        files = [filename]
        print('files: ', files)
        return render_template('index.html', pred=classnames[class_], files = files)
    return render_template('index.html', pred='')

@app.route('/api', methods=['POST'])
def api_predict():
    print("Upload - ", end='')
    image_size = app.config['IMAGE_SIZE']
    uploaded_file = request.files['image_uploads']
    filename = secure_filename(uploaded_file.filename).lower()
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
        classnames = app.config['CLASS']
        #files = getfiles(app.config['UPLOAD_PATH'])
        print('Prediction: ', classnames[class_],)
        result = {'prediction': classnames[class_]}
        json_resp = json.dumps(result)
        return json_resp
    return 

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

def main():
    """Run the app."""
    app.run(host='0.0.0.0', debug=False)  # nosec


if __name__ == '__main__':
    main()