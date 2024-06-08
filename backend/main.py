# from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
from PIL import Image
import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'backend\models\hand_landmarker.task'

# alexnet = load_model('./models/alexnet.h5')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

app = Flask(__name__)
CORS(app)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def camera_max():
    return 1


with open('./models/best_svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

cam_max = camera_max()
cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    results = model.process(image)                 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def sign_frame():  # generate frame by frame from camera
    global easy, cap
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while True:
            success, frame = cap.read() 
            if success:
                try:
                    height, width, _ = frame.shape
                    if height > width:
                        diff = (height - width) // 2
                        frame = frame[diff:height - diff, 0:width]
                    else:
                        diff = (width - height) // 2
                        frame = frame[0:height, diff:width - diff]

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(cv2.resize(frame_rgb,(224,224)))

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            x_landmark = [landmark.x for landmark in hand_landmarks.landmark]
                            y_landmark = [landmark.y for landmark in hand_landmarks.landmark]
                            x_min = min(x_landmark)
                            x_max = max(x_landmark)
                            y_min = min(y_landmark)
                            y_max = max(y_landmark)
                            
                            x_min = int(x_min * width)
                            x_max = int(x_max * width)
                            y_min = int(y_min * height)
                            y_max = int(y_max * height)
                            
                            cv2.rectangle(frame, (x_min-80, y_min-50), (x_max-20, y_max+50), (0, 255, 0), 2)

                            x_min_adj = max(0, x_min - 80)
                            x_max_adj = min(width, x_max)
                            y_min_adj = max(0, y_min - 50)
                            y_max_adj = min(height, y_max + 50)
                            roi = frame[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
                            hand_data = {}
                            for idx1, landmark in enumerate(hand_landmarks.landmark):
                                # print("x",landmark.x)
                                # print("0",hand_landmarks.landmark[0].x)
                                if(idx1==0):continue
                                relative_x = landmark.x - hand_landmarks.landmark[0].x
                                relative_y = landmark.y - hand_landmarks.landmark[0].y
                                # print(type(idx1), type(relative_x),type(relative_y))
                                hand_data[f"Landmark_{idx1 + 1}_X"] = [relative_x]
                                hand_data[f"Landmark_{idx1 + 1}_Y"] = [relative_y]
                            # roi_preprocessed = preProcess(roi)
                            # roi_preprocessed = np.expand_dims(roi_preprocessed, axis=0)

                            # plt.imshow(roi_preprocessed.squeeze(), cmap='gray')
                            # plt.title('Preprocessed Image')
                            # plt.show()
                            # plt.close()
                            # relative_x = landmark.x - hand_landmarks.landmark[0].x
                            # relative_y = landmark.y - hand_landmarks.landmark[0].y

                            target_size = (224, 224)
                            # print(roi_preprocessed)
                            # prediction = alexnet.predict(roi_preprocessed)
                            # print(hand_data)
                            prediction = model.predict(pd.DataFrame(hand_data))
                            # print(prediction)
                            # predicted_class = np.argmax(prediction)
                            cv2.putText(frame, f'Class: {prediction[0]}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(e)
                    pass

@app.route('/video_feed')
def video_feed():
    print("hehe")
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def hello():
    return render_template('index.html', person="Hendi")

if __name__ == '__main__':
    print("run")
    app.run(port=6969, debug=True)
