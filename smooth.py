import cv2
from deepface import DeepFace
import os
from datetime import datetime
import numpy as np
import msvcrt

SIMILARITY_THRESHOLD = 0.5

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_faces_dir = "known_faces"
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)


def get_next_person_id():
    existing_folders = [f for f in os.listdir(known_faces_dir)
                        if f.startswith("person_") and os.path.isdir(os.path.join(known_faces_dir, f))]
    if not existing_folders:
        return 0
    numbers = [int(f.split('_')[1]) for f in existing_folders]
    return max(numbers) + 1


def save_face(face_img, identity):
    path = os.path.join(known_faces_dir, identity)
    if not os.path.exists(path):
        os.makedirs(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(path, f"{timestamp}.jpg")
        cv2.imwrite(filename, face_img)    


def find_match(face_img):
    if len(os.listdir(known_faces_dir)) == 0:
        return None

    try:
        for person_dir in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_dir)
            if os.path.isdir(person_path) and os.listdir(person_path):
                # could make faster by just checking one
                # also could save similarity for each person and return the best one
                # Try each image in person's folder
                for img_name in os.listdir(person_path):
                    reference_img = os.path.join(person_path, img_name)
                    result = DeepFace.verify(
                        face_img,
                        reference_img,
                        enforce_detection=False,
                        distance_metric="cosine",
                        model_name="VGG-Face",
                        detector_backend="opencv",
                        threshold=SIMILARITY_THRESHOLD
                    )
                    if result['verified']:
                        return person_dir
    except:
        return None
    return None


def process_frame(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Find identity match
        identity = find_match(face_img)
        if identity is None:
            new_id = get_next_person_id()
            identity = f"person_{new_id}"

        save_face(face_img, identity)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, identity, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame


# Capture video from webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow('Face Recognition', frame)

    if msvcrt.kbhit():
        if msvcrt.getch() == b'q':
            break

video_capture.release()
cv2.destroyAllWindows()
