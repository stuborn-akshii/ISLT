from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from mediapipe.python.solutions import holistic as mp_holistic
import cv2
import tensorflow as tf
import time
import keras
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense ,LSTM, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import itertools
from PIL import Image
# from functions import *
import pickle
from flask_navigation import Navigation

app = Flask(__name__)
nav = Navigation(app)




# Page title
@app.route('/')
@app.route('/home')
def home():   
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/translate', methods=['GET'])
def translate():
    return render_template('translate.html')

@app.route('/learn', methods=['GET'])
def learn():
    return render_template('learnislt.html')

@app.route('/isltfewwords', methods=['GET'])
def isltfewwords():
    return render_template('isltfewwords.html')

# Prediction page
@app.route('/', methods=['GET', 'POST'])
def upload_video():
    msg="video not uploaded try again"
    if request.method == 'POST':
        # Handle video upload here
        video = request.files['video']
        # Save the video to the desired location
        video.save(r'C:\Users\Shatakshi\Downloads\ISLT\Videos\test1.mp4')
        
        msg="Video uploaded"
        
        model_path=r"C:\Users\Shatakshi\Downloads\ISLT\BIGRU_BILSTM_e150_tr100_val90_pa82.h5"
        model = load_model(model_path)
        # "C:\Users\Shatakshi\Downloads\ISLT WEBSITE\Hackathon_Hare_Krishna\Videos"
        output=predict_video("test",r'C:\Users\Shatakshi\Downloads\ISLT\Videos\test1.mp4',model)
        # Update with the appropriate URL
        return render_template('translate.html',mssg=msg, output=output)
    return render_template('translate.html',mssg=msg)

def predict_video(video_name,video_path,model):
  col=np.arange(225).tolist()
  track=0
  col.extend(["frame_number","video_name","output_class"])
  df_predict=pd.DataFrame(columns=col)
  df_predict,track=storeInDataFrame(r'C:\Users\Shatakshi\Downloads\ISLT\Videos',video_name,df_predict,track,video_path)
  df_predict.info()
  ds_x_predict=df_predict.iloc[:,:-3] #last 3 columns have frame no, video name and video class
  print(ds_x_predict.shape)
  ds_x_predict=np.array(ds_x_predict)
  # Normalize the prediction data that has been passed
  x_predict=MinMaxScaler().fit_transform(ds_x_predict,y=None)
  start=5
  end=5
  x_predict_del=np.delete(x_predict,np.s_[:start],axis=0)
  x_predict_del=np.delete(x_predict_del,np.s_[-end:],axis=0)
  # x_predict_del.shape
  x_predict=x_predict_del
  y_pred_probs = model.predict([np.expand_dims(x_predict,axis=0),np.expand_dims(x_predict,axis=0)])
  #method for label encoding
  classes = ['Afternoon', 'Deaf', 'He', 'Help','Morning','Thankyou','Today','Winter']
#   y_pred_probs = y_pred_probs[0]
  predicted_class_index = np.argmax(y_pred_probs)
  
  predicted_class_label = classes[predicted_class_index]
  
  
  #otherwise rarely
#   label_encoder="define"#definee
#   y_pred = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))

  # y_pred=label_encoder.inverse_transform(model.predict_classes(np.expand_dims(x_predict,axis=0)))
  # y_pred=np.argmax(np.expand_dims(x_predict,axis=0),axis=-1)
  
  return predicted_class_label

def storeInDataFrame(folder, video,df,track, path):
    data = getFromVideo(folder, video, path)
     
    for row in data:
        # print('Row shape:', len(row))
        # print('DataFrame shape:', df.shape)
        df.loc[track] = row
        track += 1
    return df, track
    

def getFromVideo(folder, video, path):
    num_frames = 18
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = total_frames // num_frames
    count = 0
    data = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            if (count + 1) % skip == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                if results.pose_landmarks is None:
                  print("none")
                  continue
                input_features = []
                # Pose landmarks
                # print("pose")
                for landmark in results.pose_landmarks.landmark:
                    input_features.extend([landmark.x, landmark.y, landmark.z])
                   

                # Left hand landmarks
                # print("left")
                if results.left_hand_landmarks is None:
                  print("none")
                  input_features.extend([0] * 63)
                else:
                  for landmark in results.left_hand_landmarks.landmark:
                      input_features.extend([landmark.x, landmark.y, landmark.z])
                      
                # Right hand landmarks
                # print("right")
                if results.right_hand_landmarks is None:
                  print("none")
                  input_features.extend([0] * 63)

                else:
                  for landmark in results.right_hand_landmarks.landmark:
                      input_features.extend([landmark.x, landmark.y, landmark.z])
                      
                input_features.extend([count, video, folder])
                # print(input_features)
                data.append(input_features)
                if len(data) == num_frames:
                    break
            count += 1
        cap.release()
    return data
        

if __name__ == '__main__':
    app.run(debug=True)
