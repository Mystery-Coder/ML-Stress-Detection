import os
import pickle
from pydub import AudioSegment
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
import joblib

#This Code only perform the output class extraction form the chunks of Audio signal
# there is no split opeation is performed !
#2->2min files and 3-> 3sec files
folder2 = "C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp\\audiofile-step3-2"
folder3 = "C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp\\audiofile-step3-3"

emotion = []
depression = []


#I am Loadig file ehere
lb_dp = pickle.load(open("lb\\lb-depression.sav", 'rb'))
lb_emo = pickle.load(open("lb\\lb-emotion.sav", 'rb'))

model_emotion = load_model("Model\\emotion.keras")
model_depression = load_model("Model\\depression.keras")

def extract_mfcc(file_path, duration, sr, offset, n_mfcc):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=duration, sr=sr, offset=offset)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc), axis=0)
    return mfccs

def prepare_features(mfcc_features, required_length):
    current_length = len(mfcc_features)
    if current_length < required_length:
        padded_array = np.pad(mfcc_features, (0, required_length - current_length), 'constant', constant_values=0)
    else:
        padded_array = mfcc_features[:required_length]
    return np.expand_dims(padded_array, axis=0)


#Emotion part here
emo_duration = 3 

for i, file in enumerate(os.listdir(folder3)):
    file_path = os.path.join(folder3, file)
    mfccs = extract_mfcc(file_path, emo_duration,44100,0.5,13)
    x_testcnn = prepare_features(mfccs,259)

    y_pred = model_emotion.predict(x_testcnn)
    predicted_class = np.argmax(y_pred, axis=1)  
    predicted_emotion = lb_emo.inverse_transform(predicted_class)  
    emotion.append(predicted_emotion[0]) 
     
    


#Depression part here
dep_duration = 2 * 60  
for i, file in enumerate(os.listdir(folder2)):
    file_path = os.path.join(folder2, file)
    mfccs = extract_mfcc(file_path, dep_duration,44100,0.5,20)
    x_testcnn = prepare_features(mfccs ,10293)

    # Predict depression
    y_pred = model_depression.predict(x_testcnn)
    predicted_class = np.argmax(y_pred, axis=1)  
    predicted_depression = lb_dp.inverse_transform(predicted_class)  
    depression.append(predicted_depression[0])
    

data={
    "depression":depression,
    "emotion":emotion
}
with open("predictions.json", "w") as json_file:
        json.dump(data, json_file, indent=4)



