from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
from PIL import Image
import io
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'backend\models\hand_landmarker.task'

alexnet = load_model('./models/alexnet.h5')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

app = Flask(__name__)
CORS(app)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def camera_max():
    return 1
       
cam_max = camera_max()
cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)


def preProcess(image):
    gray = tf.image.rgb_to_grayscale(image)
    gray = tf.squeeze(gray, axis=-1)

    gray_np = tf.image.convert_image_dtype(gray, tf.uint8).numpy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cequ_hist = clahe.apply(gray_np)

    gauss_blur = cv2.GaussianBlur(cequ_hist, (3, 3), 20)

    # sobel_x = cv2.Sobel(gauss_blur, cv2.CV_32F, 1, 0, 3)
    # sobel_y = cv2.Sobel(gauss_blur, cv2.CV_32F, 0, 1, 3)

    # merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # merged_sobel *= 255 / merged_sobel.max()

    resized_image = cv2.resize(gauss_blur, (224, 224))

    resized_image = np.expand_dims(resized_image, axis=-1)

    return resized_image


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
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                            y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                            
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
                            
                            roi_preprocessed = preProcess(roi)
                            roi_preprocessed = np.expand_dims(roi_preprocessed, axis=0)

                            # plt.imshow(roi_preprocessed.squeeze(), cmap='gray')
                            # plt.title('Preprocessed Image')
                            # plt.show()
                            # plt.close()

                            target_size = (224, 224)
                            # print(roi_preprocessed)
                            prediction = alexnet.predict(roi_preprocessed)
                            predicted_class = np.argmax(prediction,axis=1)
                            cv2.putText(frame, f'Class: {classes[predicted_class[0]]}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
