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
import requests

app = Flask(__name__)
app.config['MODEL_PATH'] = 'EfficientNet_CL_20210312'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', 'jpeg']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['IMAGE_SIZE'] = 224
app.config['CLASS'] =  ['Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy' ,'Apple scab',
 'Bell Pepper Bacterial spot' ,'Bell Pepper healthy', 'Blueberry healthy',
 'Cherry (including sour) Powdery mildew',
 'Cherry (including sour) healthy',
 'Corn (maize) Cercospora leaf spot Gray leaf spot',
 'Corn (maize) Common rust ' ,'Corn (maize) Northern Leaf Blight',
 'Corn (maize) healthy' ,'Grape Black rot', 'Grape Esca (Black Measles)',
 'Grape Leaf blight (Isariopsis Leaf Spot)' ,'Grape healthy',
 'Orange Haunglongbing (Citrus greening)' ,'Peach Bacterial spot',
 'Peach healthy', 'Potato Early blight', 'Potato Late blight',
 'Potato healthy', 'Raspberry healthy' ,'Soybean healthy',
 'Squash Powdery mildew' ,'Strawberry Leaf scorch' ,'Strawberry healthy',
 'Tomato American Serpentine Leafminer', 'Tomato Bacterial spot',
 'Tomato Early blight' ,'Tomato Insect Bite' ,'Tomato Late blight',
 'Tomato Leaf Mold', 'Tomato Powdery mildew', 'Tomato Septoria leaf spot',
 'Tomato Spider mites Two-spotted spider mite', 'Tomato Stem rot',
 'Tomato Target Spot' ,'Tomato Wilt', 'Tomato Yellow Leaf Curl Virus',
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
    return render_template('qna2.html')

@app.route("/index2")
def qna2():
    #files = os.listdir(app.config['UPLOAD_PATH'])
    #print('files: ', files)
    return render_template('uploader/index.html')


def getQnaAnswer(question):
    url = 'https://ichat.azurewebsites.net/qnamaker/knowledgebases/8f09b2a4-96db-4402-abb7-d38eb9ae9cd4/generateAnswer'
    headers = {'Authorization': 'EndpointKey bd260132-8cb3-457c-9451-87e4bc26f7fa', 'content-type': 'application/json'}
    json_data = "{'question':'What is {" + question + "}?'}"
    response = requests.post(url, data=json_data, headers=headers)
    answer = json.loads(response.text)['answers'][0]['answer']
    return answer

def getResultAndAnswer(img):
    image_size = app.config['IMAGE_SIZE']
    predictions = app.predictor.predict(img.reshape(-1,image_size,image_size,3))
    confidence = tf.nn.softmax(predictions)
    #print('INFO Predictions: {}'.format(predictions))
    print('INFO confidence: {}'.format(confidence))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
    classnames = app.config['CLASS']
    print(class_)
    #print(answer)  
    result = f"{classnames[class_]} ({confidence[0,class_]*100:1.1f}%)"    
    answer = getQnaAnswer(classnames[class_])
    
    return result, answer

def getScore(img, newline=True):
    result, answer = getResultAndAnswer(img)
    if newline:
        result += f"\n\n{answer}"
    else:
        result += f": {answer}"
    return result

@app.route('/score', methods=['POST'])
def score():
    image_size = app.config['IMAGE_SIZE']
    print(request)
    nparr = np.frombuffer(request.data, np.uint8)
    print(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
    
    predictions = app.predictor.predict(img.reshape(-1,image_size,image_size,3))
    confidence = tf.nn.softmax(predictions)
    #print('INFO Predictions: {}'.format(predictions))
    print('INFO confidence: {}'.format(confidence))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
    classnames = app.config['CLASS']
    print(class_)
    #print(answer)  
    result = f"{classnames[class_]} ({confidence[0,class_]*100:1.1f}%)"    
    #answer = getQnaAnswer(classnames[class_])
    
    return result


@app.route('/predict', methods=['POST'])
def predict():
    image_size = app.config['IMAGE_SIZE']
    print(request)
    nparr = np.frombuffer(request.data, np.uint8)
    print(nparr)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size,image_size), interpolation = cv2.INTER_AREA)
    
    #cdcv2.imwrite('test.jpg',img)
    return getScore(img)

#this is returns a html page with prediction
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
        pred = getScore(img, False)
        
        #files = getfiles(app.config['UPLOAD_PATH'])
        files = [filename]
        print('files: ', files)
        return render_template('index.html', pred=pred, files = files, scroll="diagnosis")
    return render_template('index.html', pred='', scroll="diagnosis")


#this function returns score for POST request with file
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
        pred = getScore(img, False)
        #files = getfiles(app.config['UPLOAD_PATH'])
        #result = {'prediction': pred}
        json_resp = json.dumps(pred)
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
