import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow import keras

# 1. New detection variables


mp_hands = mp.solutions.hands # Hands model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    hand = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand = np.array(np.array([[res.x, res.y, res.z, res.visibility] for res in hand_landmarks.landmark]).flatten())
            hand = np.expand_dims(hand, 0)
    else:
        hand = np.zeros((1,84))
    return hand

prev = None
class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.threshold = 0.7
        self.actions = "a b c d e f g h i k l m n o p q r s t u v w x y".split(" ")
        self.actions = np.array(self.actions)
        inputs = tf.keras.Input(shape=(84))
        layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        outputs = keras.layers.Dense(len(self.actions), activation="softmax")(layer)
        self.model = keras.Model(inputs, outputs)
        self.model.load_weights('version1.h5')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Set mediapipe model 
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            global prev
            while self.video.isOpened():
                # Read feed
                ret, frame = self.video.read()

                # ignore empty frame
                if not ret:
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                keypoints = extract_keypoints(results)
                res = self.model.predict(keypoints)[0]
                
                if res[np.argmax(res)] > 0.6:
                    if prev != self.actions[np.argmax(res)]:
                        prev = self.actions[np.argmax(res)]
                        
                cv2.putText(image, f"Classified: {prev}",(15,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)
                cv2.putText(image, "Confidence: {:.4f}".format(res[np.argmax(res)]),(15,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)
                cv2.imshow('Mediapipe hands', image)
                
                return cv2.imencode(".jpg",image)[1].tobytes()


