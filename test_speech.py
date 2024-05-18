#importing ncessary libraries
import pandas as pd
import numpy as np
import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Audio
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
#from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pickle

#loading standardscaler
scaler=pickle.load(open('Project_Saved_Models/speech_scaler.pkl','rb'))

#extract features
def _extract__Features_(_data,sample_rate):
    # Zeor crossing rate
    _result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=_data).T, axis=0)
    _result=np.hstack((_result, zcr)) 

    # Chroma
    stft = np.abs(librosa.stft(_data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, chroma_stft)) 

   #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mfcc)) 

    # RMS Value
    rms = np.mean(librosa.feature.rms(y=_data).T, axis=0)
    _result = np.hstack((_result, rms))

   #melspectogram
    mel = np.mean(librosa.feature.melspectrogram(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mel)) # stacking horizontally
    
    return _result

def _get__Features_(path):
    
    _data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print(path)
    
   
    res1 = _extract__Features_(_data,sample_rate)
    _result = np.array(res1)
    
    return _result

def shift(_data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(_data, shift_range)

def pitch(_data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(_data, sampling_rate, pitch_factor)

#load the trained model
model=load_model('Project_Saved_Models/trained_speech_model.h5')


if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    #feature extaction
    feat=_get__Features_(path)
    feat=np.array([feat])
    #perform standardization
    feat=scaler.transform(feat)
    #expand dimension
    feat = np.expand_dims(feat, axis=2)
   
   #prediction using the model
    pred=model.predict(feat)[0]
    # print(pred)
    pred=np.argmax(pred)
    # print(pred)

    if pred==0:
        print("Depressive")
    elif pred==1:
        print("Non-Depressive")
