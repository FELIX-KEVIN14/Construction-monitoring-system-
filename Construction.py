import streamlit as st
import cv2
import pickle
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from datetime import timedelta, datetime
import pandas as pd
import pygame
import csv

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
project_folder = os.path.dirname(os.path.abspath(__file__))
names_file = os.path.join(project_folder, 'names.pkl')
mugam_data_file = os.path.join(project_folder, 'mugam_data.pkl')
attendance_folder = os.path.join(project_folder, 'attendance_csv')
emotions = ["Sad", "Happy", "Cry", "Neutral"]
def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    # Extract coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of each bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their confidence scores in descending order
    idxs = np.argsort(boxes[:, 4])[::-1]

    # Initialize the list to store the final bounding boxes
    pick = []

    while len(idxs) > 0:
        # Get the index of the current bounding box with the highest confidence
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the coordinates of the intersection rectangle
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the intersection rectangle
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the specified threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return pick

def object_monitor(url):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getUnconnectedOutLayersNames()

    # Output CSV file
    csv_filename = f"object_counts_{datetime.now().strftime('%Y-%m-%d')}.csv"

    # Check if the file already exists
    file_exists = os.path.exists(csv_filename)

    # Open the CSV file in append mode if it exists, otherwise create a new file
    csv_file = open(csv_filename, 'a' if file_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # If the file is newly created, write the header
    if not file_exists:
        csv_writer.writerow(['timestamp', 'object_count'])

    # Time interval for counting and saving data (in seconds)
    time_interval = 4

    # Initialize start time
    start_time = time.time()

    # Initialize the camera capture
    cap = cv2.VideoCapture(url)

    # Main loop for capturing and processing video
    video_placeholder = st.empty()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Resize frame and normalize it
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Perform forward pass and get predictions
        outputs = net.forward(layer_names)

        # Threshold for confidence
        confidence_threshold = 0.5

        # List to store current bounding boxes
        current_boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold and class_id == 0:  # Class ID for person is 0
                    center_x, center_y, w, h = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                    x, y = int(center_x - w/2), int(center_y - h/2)

                    current_boxes.append([x, y, x+w, y+h, confidence])

        # Apply non-maximum suppression to remove overlapping boxes
        indices = non_max_suppression(np.array(current_boxes), overlap_thresh=0.3)

        # Count persons
        person_count = len(indices)

        # Display the result
        cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with bounding boxes (optional)
        for index in indices:
            x, y, x2, y2, _ = current_boxes[index]
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Display the image with bounding boxes using video_placeholder
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check if the time interval has passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_interval:
            # Save timestamp and object count to CSV file
            timestamp = datetime.now().strftime('%H:%M:%S')
            csv_writer.writerow([timestamp, person_count])

            # Reset start time
            start_time = time.time()

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera, close CSV file, and close all windows
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def register_face(video_placeholder, mugam_detect, url, names_file, mugam_data_file):
    mugam_data = []
    i = 0

    # Streamlit input for name
    name = st.text_input('Enter the name: ')

    # Start face registration when the "Start Registration" button is clicked
    if st.button("Start Registration", key="start_registration"):
        # Initialize the camera capture
        cap = cv2.VideoCapture(url)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in mugam:
                # Crop the face region
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))

                # Capture every 10th frame (adjust as needed)
                if len(mugam_data) < 100 and i % 10 == 0:
                    mugam_data.append(resized_img)

                i += 1

                # Display count on the frame
                cv2.putText(frame, str(len(mugam_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            # Display the image with bounding boxes using video_placeholder
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Check if 100 frames are captured
            if len(mugam_data) == 100:
                break

        # Release the camera
        cap.release()

        # Convert mugam_data to NumPy array
        mugam_data = np.asarray(mugam_data)
        mugam_data = mugam_data.reshape(100, -1)

        # Save or update names_file
        if not os.path.exists(names_file):
            names = [name] * 100
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)
        else:
            with open(names_file, 'rb') as f:
                names = pickle.load(f)
            names = names + [name] * 100
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)

        # Save or update mugam_data_file
        if not os.path.exists(mugam_data_file):
            with open(mugam_data_file, 'wb') as f:
                pickle.dump(mugam_data, f)
        else:
            with open(mugam_data_file, 'rb') as f:
                mugam = pickle.load(f)
            mugam = np.append(mugam, mugam_data, axis=0)
            with open(mugam_data_file, 'wb') as f:
                pickle.dump(mugam, f)
    
def attendance_monitoring(url):
    video = cv2.VideoCapture(url)
    mugam_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    today_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"detections_{today_date}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Class", "Confidence", "X", "Y", "Width", "Height"])
    unique_detections = {}

    person_detected = False
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    with open(names_file, 'rb') as f:
        labels = pickle.load(f)
    with open(mugam_data_file, 'rb') as f:
        MUGAM = pickle.load(f)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(MUGAM, labels)
    col = ['Name', 'Present', 'Time', 'Drowsiness', "Emotion"]
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    video_placeholder = st.empty()

    # Initialize attendance_file outside the loop
    attendance_file = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        output_layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(output_layer_names)
        current_time = datetime.now().strftime("%H:%M:%S")
        person_detected = False

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    key = (classes[class_id], x, y, w, h)

                    if key not in unique_detections:
                        csv_writer.writerow([current_time, classes[class_id], confidence, x, y, w, h])
                        color = (0, 255, 0)
                        text_color = (255, 255, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{classes[class_id]}: {confidence:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, cv2.FILLED)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                        unique_detections[key] = True

                    if classes[class_id] == "person":
                        person_detected = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        faces = mugam_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in mugam:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            face_roi = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            emotion = emotions[3]

            if len(eyes) >= 2:
                eye_region = face_roi[eyes[0][1]:eyes[0][1] + eyes[0][3], eyes[0][0]:eyes[0][0] + eyes[0][2]]
                avg_intensity = cv2.mean(eye_region)[0]

                if avg_intensity < 60:
                    emotion = emotions[0] if avg_intensity < 40 else emotions[2]
                elif avg_intensity > 100:
                    emotion = emotions[1]

            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = shape_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            COUNTER = 0
            TOTAL = 0
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            drowsiness = "Yes" if COUNTER >= EYE_AR_CONSEC_FRAMES else "No"
            
            # Update attendance_file only if faces are detected
            if len(mugam) > 0:
                attendance_file = os.path.join(attendance_folder, 'attendance_' + date + '.csv')
                exist = os.path.isfile(attendance_file)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (255, 0, 0), -1)
                cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(frame, "Drowsiness: {}".format(drowsiness), (x, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

                video_placeholder.image(frame, channels="BGR", use_column_width=True)

                with open(attendance_file, 'a') as f:
                    if not exist:
                        f.write(','.join(col) + '\n')
                    f.write(','.join([str(output[0]), 'Present', str(timestamp), drowsiness, emotion]) + '\n')

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
def main():
    
    st.title("Object Monitoring and Attendance System")

    # Sidebar for selecting camera source
    camera_source = st.sidebar.radio("Select Camera Source", ["System Camera", "Camera IP"])
    if camera_source == "Camera IP":
        url = st.sidebar.text_input("Enter Camera IP Address:")
    else:
        url = 0  # Default to system camera

    # Sidebar for selecting actions
    action = st.sidebar.radio("Select Action", ["Take Attendance", "Store Face Data", "Monitor Resources", "Exit"])

    if action == "Take Attendance":
        if st.button("Start"):
            st.write("Capturing video...")
            attendance_monitoring(url)

    elif action == "Store Face Data":
        names_file = "names.pkl"
        mugam_data_file = "mugam_data.pkl"

        # Load the face detection cascade classifier
        mugam_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Create a video placeholder
        video_placeholder = st.empty()

        # Call the register_face function
        register_face(video_placeholder, mugam_detect, url, names_file, mugam_data_file)

    elif action == "Monitor Resources":
        if st.button("Start"):
            st.write("Capturing video for resource monitoring...")
            object_monitor(url)

    elif action == "Exit":
        st.write("Exiting the application...")

    # ... (remaining code for other actions)


if __name__ == "__main__":
    main()
