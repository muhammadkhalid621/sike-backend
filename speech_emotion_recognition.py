import IPython.display as ipd
import pickle
import numpy as np # linear algebra
import librosa
from tensorflow.keras.models import model_from_json

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name,duration=3
                                  ,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs

def predict_speech_emotion():
    json = open('model.json', 'r').read()
    model= model_from_json(json)

    # load weights into new model
    model.load_weights('speech_emotion.h5')

    path_ = '03-01-04-02-01-01-03.wav'
    ipd.Audio(path_)
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    q = np.expand_dims(a1,-1)

    qq = np.expand_dims(q,0)

    pred = model.predict(qq)

    preds=pred.argmax(axis=1)
    if isinstance(preds[0], np.int64):
        return int(preds[0])
    return preds[0]
