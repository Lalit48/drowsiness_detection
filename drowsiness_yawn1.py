from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
from twilio.rest import Client  # Twilio for SMS alerts

# Twilio Configuration (Replace with actual credentials)
ACCOUNT_SID = ""
AUTH_TOKEN = ""
TWILIO_NUMBER = ""
RECIPIENT_NUMBER = ""

# Alarm sound function
def sound_alarm(path):
    playsound.playsound(path)

# Send SMS alert
def send_sms_alert(msg):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    message = client.messages.create(body=msg, from_=TWILIO_NUMBER, to=RECIPIENT_NUMBER)
    print("SMS Alert Sent! Message SID:", message.sid)

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Get final EAR for both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

# Lip distance for yawn detection
def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Webcam index")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 90  # 3 seconds at 30 FPS
YAWN_THRESH = 20
YAWN_LIMIT = 2  # Send alert after 2 yawns
alarm_status = False
sms_sent = False
COUNTER = 0
YAWN_COUNTER = 0

# Load Face Detector & Predictor
print("-> Loading Face Detector & Predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # Drowsiness detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not sms_sent:
                print("DROWSINESS ALERT!")
                Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
                Thread(target=send_sms_alert, args=("Drowsiness detected! Please take a break.",), daemon=True).start()
                sms_sent = True
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            sms_sent = False

        # Yawn detection
        if distance > YAWN_THRESH:
            YAWN_COUNTER += 1
            cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if YAWN_COUNTER > YAWN_LIMIT and not sms_sent:
                print("YAWN ALERT!")
                Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
                Thread(target=send_sms_alert, args=("Excessive yawning detected! Take a break.",), daemon=True).start()
                sms_sent = True
            YAWN_COUNTER = 0

        # Display EAR & YAWN metrics
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
