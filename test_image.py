import numpy as np
import cv2
import os
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten,Dropout, Dense,Convolution2D,MaxPooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tkinter.filedialog import askopenfilename

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

model.load_weights('Project_Saved_Models/trained_image_model.h5')
# loaded_model = load_model("Project_Saved_Models/trained_image_model.h5")
width = 48
height = 48

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Depression", 4: "Surprised"}
def prediction(path):
    image=cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(image, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img) 
    maxindex = int(np.argmax(prediction))
    # print(maxindex)
    print("\n********Result********")
    print(emotion_dict[maxindex])
    

if __name__=='__main__':
    path=askopenfilename()
    prediction(path)