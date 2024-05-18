import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
import string
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')#tokenize the text in the dataset.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from nltk import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import string
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score



#Load the trained model & Vectorizer
tokenizer = pickle.load(open('Project_Saved_Models/text_tokenizer.pkl', 'rb'))
loaded_model=load_model('Project_Saved_Models/trained_text_model.h5')

max_length=100

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
    stop_words = list(stopwords.words("english"))
    stop_words.remove('not')
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


if __name__=="__main__":

	text=input("Enter a text : ")

	a=convert_to_lower(text)
	b=remove_numbers(a)
	c=remove_punctuation(b)
	d=remove_stopwords(c)
	e=remove_extra_white_spaces(d)
	f=lemmatizing(e)

	data=[f]
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
	else:
		print("Non Depressive")

	# pred_labels = []
	# for i in prediction:
	#     if i >= 0.5:
	#         pred_labels.append(1)
	#     else:
	#         pred_labels.append(0)
	# print(pred_labels)
	# for i in range(len(data)):
	#     # print(data[i])
	#     if pred_labels[i] == 1:
	#         s = 'Depressive'
	#     else:
	#         s = 'Non Depressive'
	#     print("Predicted sentiment : ",s)



