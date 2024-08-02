# drowsiness_detection

### Key Components

1. **Libraries and Modules:**
   - `imutils`, `dlib`, `cv2`: For image processing and facial landmark detection.
   - `scipy.spatial.distance`: To calculate the Euclidean distance between points.
   - `os`, `time`, `threading`, `json`: For file handling, timing, threading, and data handling.
   - `twilio.rest.Client`: To send SMS alerts.
   - `sounddevice`: For playing audio alerts.
   - `numpy`: For numerical computations.

2. **Twilio Integration:**
   - Used to send SMS alerts to a specified phone number if drowsiness is detected.

3. **Functions:**
   - `eye_aspect_ratio(eye)`: Computes the Eye Aspect Ratio (EAR) to determine if the eyes are closed.
   - `send_sms_alert(message)`: Sends an SMS alert via Twilio.
   - `play_sound(file_path)`: Plays a sound file as an alert.
   - `alert()`: Plays an audio alert and sends an SMS alert.

4. **Parameters:**
   - `thresh`: EAR threshold below which the eyes are considered closed.
   - `eye_closed_time`: Time in seconds that eyes must be closed before triggering an alert.

5. **Paths and Model Loading:**
   - `shape_predictor_path`: Path to the dlib shape predictor model file.
   - Verifies that the shape predictor file exists.

6. **Main Loop:**
   - Captures video frames using `cv2.VideoCapture(0)`.
   - Converts the frame to grayscale and detects faces.
   - For each detected face, it calculates the EAR for both eyes.
   - If the EAR falls below the threshold, it checks the duration for which the eyes have been closed.
   - If eyes remain closed for more than the specified `eye_closed_time`, an alert is triggered.

### Workflow
1. The system initializes and starts capturing video from the webcam.
2. It continuously processes the frames to detect faces and eyes.
3. The EAR is calculated for both eyes to determine if the driver is potentially drowsy.
4. If drowsiness is detected (eyes closed for too long), the system plays a sound and sends an SMS alert.

### Improvements and Considerations
- **Security:** Ensure that sensitive information like Twilio credentials is stored securely and not hard-coded.
- **Error Handling:** Handle potential errors, such as missing shape predictor files or failed SMS sends.
- **Optimization:** Consider optimizing the alerting system to prevent false positives and ensure timely alerts.

