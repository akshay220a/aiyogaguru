from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/workout', methods=['GET', 'POST'])
def workout():
    return render_template('workoutcam.html')








mp_draw = mp.solutions.drawing_utils  
mp_pose = mp.solutions.pose 
confidence_threshold = 0.85

model = load_model('model/yoga_model9.h5')

actions=['Adha Mukha Svanasana', 'Bitilasana', 'Eka Pada Adha Mukha Svanasa',
 'Padmasana', 'Trikon Asana','Unknown Pose', 'Virbhadra-1' ,'Virbhadra-2' ,'Vrikshasana']

# Define the indices of landmarks you want to extract coordinates for
landmark_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]






@app.route('/process_data', methods=['POST'])
def predict():

    data = request.json
    pose_landmarks = data.get('poseLandmarks')
    additional_data = data.get('additionalData')

    pose_row = [] 

    for idx, landmark in enumerate(pose_landmarks):
        if idx in landmark_indices:
            pose_row.extend([landmark['x'], landmark['y'], landmark['z'], landmark['visibility']])


    pose_row = np.array(pose_row)

    x = pd.DataFrame([pose_row])
    body_language_prob = model.predict(x)[0]
    body_language_class = actions[np.argmax(body_language_prob)]

    if (np.max(body_language_prob) < confidence_threshold) or (body_language_class!=additional_data):
        body_language_class = "Unknown Pose"

    return body_language_class


