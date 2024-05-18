# import necessary libraries
from django.shortcuts import render
import json
from django.core import serializers
from .models import *
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.db.models import Count
from os import path
from pydub import AudioSegment
import re
from django.views.decorators.cache import never_cache
from django.core.files.storage import FileSystemStorage
import base64
import cv2
import os
import numpy as np
import random
from datetime import datetime
from datetime import date
import pandas as pd
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
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')#tokenize the text in the dataset.
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
import string
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten,Dropout, Dense,Convolution2D,MaxPooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tkinter.filedialog import askopenfilename
import random
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import math
from .stopwords import english
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from django.views.decorators.csrf import csrf_exempt



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(612, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.load_weights('DD_app/static/trained_models/trained_image_model.h5')
width = 48
height = 48

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Depression", 4: "Surprised"}

tokenizer = pickle.load(open('DD_app/static/trained_models/text_tokenizer.pkl', 'rb'))
loaded_model=load_model('DD_app/static/trained_models/trained_text_model.h5')

scaler=pickle.load(open('DD_app/static/trained_models/speech_scaler.pkl','rb'))
loaded_model1=load_model('DD_app/static/trained_models/trained_speech_model.h5')

stopwords_list=english.split('\n')

max_length=100

# Create your views here.
# @never_cache
# def display_login(request):
#     return render(request, "login.html", {})


# # def show_register(request):
# #     return render(request, "register.html", {})

# @never_cache
# def logout(request):
#     if 'uid' in request.session:
#         del request.session['uid']
#     return render(request,'login.html')





# def check_login(request):
# 	username = request.GET.get("uname")
# 	password = request.GET.get("password")

# 	print(username)
# 	print(password)

# 	d = Users.objects.filter(username=username, password=password)
# 	c = d.count()
# 	if c == 1:
# 		d2 = Users.objects.get(username=username, password=password)
# 		request.session["uid"] = d2.u_id
# 		return HttpResponse("Login Successful")
# 	else:
# 		return HttpResponse("Invalid")


# @csrf_exempt
# def find_login(request):
# 	username=request.POST.get("username")
# 	password=request.POST.get("password")

# 	print(username,password)
# 	if(username=="admin" and password=="admin"):
	    
# 	    data={"msg":"Admin"}
	 
# 	    return JsonResponse(data,safe=False)
# 	else:

#         data={"msg":"no"}
#         return JsonResponse(data,safe=False)


# @never_cache
# def show_home_user(request):
# 	if 'uid' in request.session:

# 		return render(request,'home_user.html') 
# 	else:
# 		return render(request,'login.html')


# @never_cache
# def display_check(request):
# 	if 'uid' in request.session:

# 		return render(request,'check_page.html') 
# 	else:
# 		return render(request,'login.html')







@csrf_exempt
def register(request):
	username = request.GET.get("username")
	password = request.GET.get("password")
	email=request.GET.get("email")
	phone = request.GET.get("phone")
	response_data={}
	try:
		d = Users.objects.filter(username=username)
		c = d.count()
		if c == 1:
			response_data['msg'] = "Already registered"
		else:
		    ob=Users(username=username,password=password,email=email,phone=phone)
		    ob.save()
		    response_data['msg'] = "yes"
	except:
	    response_data['msg'] = "no"
	return JsonResponse(response_data)

@csrf_exempt
def find_login(request):
	username=request.POST.get("username")
	password=request.POST.get("password")

    try:
        ob=Users.objects.get(username=username,password=password)
     
        data={"msg":"User"}
        return JsonResponse(data,safe=False)
    except:
        data={"msg":"no"}
        return JsonResponse(data,safe=False)










def predict_image(path):
    image=cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(image, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img) 
    maxindex = int(np.argmax(prediction))
    # print(maxindex)
    print("\n********Result********")
    print(emotion_dict[maxindex])
    res=emotion_dict[maxindex]

    return res

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

#Preprocessing
def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    removed = []
    stop_words = stopwords_list
    # stop_words.remove('not')
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

def predict_text(text):

	a=convert_to_lower(text)
	b=remove_numbers(a)
	c=remove_punctuation(b)
	d=remove_stopwords(c)
	e=remove_extra_white_spaces(d)
	

	data=[e]
	# convert to a sequence
	sequences = tokenizer.texts_to_sequences(data)
	# pad the sequence
	padded = pad_sequences(sequences, padding='post', maxlen=max_length)
	# Get labels based on probability 1 if p>= 0.5 else 0
	prediction = loaded_model.predict(padded)
	prediction=prediction[0]
	prediction=prediction[0]
	if prediction>=0.5:
		print("Depressive")
		s="Depressive"
	else:
		print("Non Depressive")
		s="Non Depressive"

	return s,prediction

def get_recommendation(level):
	suggested=["Get adequate sleep","Drink plenty of water","Help someone else"]
	print("the level is : ",level)
	if(level=="Level 1"):
		my_result=random.choice(suggested)
	elif(level=="Level 2"):
		my_result="Get some Excercise"
	elif(level=="Level 3"):
		my_result="Get Music Therapy"
	elif(level=="Level 4"):
		my_result="Get Counselling"
	else:
		my_result="You are safe"

	return my_result


def predict_speech(path):
	feat=_get__Features_(path)
	feat=np.array([feat])
	#perform standardization
	feat=scaler.transform(feat)
	#expand dimension
	feat = np.expand_dims(feat, axis=2)

	#prediction using the model
	pred=loaded_model1.predict(feat)[0]
	# print(pred)
	pred=np.argmax(pred)
	# print(pred)

	if pred==0:
		print("Depressive")
		get_res="Depression"
	elif pred==1:
		print("Non-Depressive")
		get_res="Non-Depressive"

	return get_res



import subprocess
	
@csrf_exempt
def predict_depression(request):

	get_text=request.POST.get("text")
	audval=request.POST.get("audval")
	image=request.POST.get("image")

	# print("\nInput")
	# print(get_text)
	# print(audval)
	# print(image)
	# print("\n")

	base64_audio_bytes = audval.encode('utf-8')
	with open('audio.mp3', 'wb') as file_to_save:
		decoded_audio_data = base64.decodebytes(base64_audio_bytes)
		file_to_save.write(decoded_audio_data)

	base64_img_bytes = image.encode('utf-8')
	with open('image.jpg', 'wb') as file_to_save:
		decoded_image_data = base64.decodebytes(base64_img_bytes)
		file_to_save.write(decoded_image_data)

	# output_file = "DD_app/static/audio.wav"

	print("\n**************")
	print(os.path.exists('audio.mp3'))
	
	# subprocess.call(['ffmpeg', '-i', 'audio.mp3','audio1.wav'])
	# print("\n**************")
	# print(os.path.exists('audio1.wav'))


	# # convert mp3 file to wav file
	# sound = AudioSegment.from_mp3('audio.mp3')
	# sound.export('audio1.wav', format="wav")
	# print("\n**************")
	# print(os.path.exists('audio1.wav'))
	# image_file= request.FILES["image_file"]
	# image_file_name=str(image_file.name)
	# print("image_file: ",image_file)
	# print("image_file_name: ",image_file_name)

	# speech_file= request.FILES["speech_file"]
	# speech_file_name=str(speech_file.name)
	# print("speech_file: ",speech_file)
	# print("speech_file_name: ",speech_file_name)

	# print(get_text)

	
	# fs1 = FileSystemStorage("DD_app/static/image_files/")
	# fs1.save(image_file.name, image_file)

	# fs2 = FileSystemStorage("DD_app/static/audio_files/")
	# fs2.save(speech_file.name, speech_file)

	# path1="DD_app/static/image_files/"+str(image_file.name)
	# path2="DD_app/static/audio_files/"+str(speech_file.name)

	out1,prediction=predict_text(get_text)
	out2=predict_image('image.jpg')
	out3=predict_speech('audio.mp3')




	if out1=="Depressive" and out2=="Depression" and out3=="Depression":
		result1="Depression Detected"
		print("Prediction value : ",prediction)
		if prediction<0.9:
			print("here1")
			level="Level 3"
		elif 0.9<=prediction<=0.99999999:
			print("here2")
			level="Level 4"

	elif((out1=="Depressive" and out2!="Depression" and out3!="Depression") or (out1=="Non Depressive" and out2=="Depression" and out3!="Depression") or (out1=="Non Depressive" and out2!="Depression" and out3=="Depression")):
		result1="Depression Detected"
		level="Level 1"

	elif((out1=="Depressive" and out2=="Depression" and out3!="Depression") or (out1=="Depressive" and out2!="Depression" and out3=="Depression") or (out1=="Non Depressive" and out2=="Depression" and out3=="Depression")):
		result1="Depression Detected"
		level="Level 2"

	else:
		result1="Safe"
		level="Safe"
		print('here3')

	out_recommend=get_recommendation(level)
	response_data={}
	if out_recommend=="You are safe":

		response_data["msg"]="Depression Not Detected"
		response_data["out1"]="Safe : Depression Not Detected"
		return JsonResponse(response_data,safe=False)
		# return HttpResponse("<script>alert('Depression Not Detected');window.location.href='/show_home_user/'</script>")
	else:
		response_data["msg"]="yes"
		response_data["out1"]=result1
		response_data["out2"]=level
		response_data["out3"]=out_recommend
		return JsonResponse(response_data,safe=False)
		# return render(request,'result_page.html',{'out1':result1,'out2':level,'out3':out_recommend})


# @never_cache
# def done(request):
#     return render(request, "check_page.html", {})