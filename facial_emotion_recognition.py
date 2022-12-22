import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

def predict_facial_emotion ():

    model_json_file = 'model _facial.json'
    model_weights_file = 'FEC_model.h5'
    with open(model_json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_file)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('test2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
        text_idx = np.argmax(pred)
        print(text_idx)
        return text_idx