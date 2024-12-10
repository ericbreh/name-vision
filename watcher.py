import cv2
from deepface import DeepFace
import os
from datetime import datetime
import time
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SIMILARITY_THRESHOLD = 0.8  # Higher = more similar
MIN_IMAGES_PER_PERSON = 1  # Minimum images to keep per person
MAX_IMAGES_PER_PERSON = 5  # Maximum images to keep per person

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


def get_image_histogram(image):
    # Convert to grayscale and calculate histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def get_uniqueness_score(target_img, other_images):
    """Calculate how unique an image is compared to others"""
    target_hist = get_image_histogram(target_img)
    total_diff = 0

    for other_img in other_images:
        other_hist = get_image_histogram(other_img)
        similarity = cv2.compareHist(
            target_hist, other_hist, cv2.HISTCMP_CORREL)
        total_diff += (1 - similarity)  # Convert similarity to difference

    return total_diff / len(other_images) if other_images else 1.0


def update_person_images(new_image, person_folder):
    """Maintain collection of most unique images"""
    images = []
    scores = []

    # Load existing images
    for img_name in os.listdir(person_folder):
        if img_name.endswith('.jpg'):
            path = os.path.join(person_folder, img_name)
            img = cv2.imread(path)
            images.append((img, path))

    # If under limit, just save
    if len(images) < MAX_IMAGES_PER_PERSON:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(person_folder, f"{timestamp}.jpg")
        cv2.imwrite(filename, new_image)
        return True

    # Calculate uniqueness scores including new image
    all_images = [img for img, _ in images] + [new_image]
    for img in all_images:
        others = [x for x in all_images if not np.array_equal(x, img)]
        score = get_uniqueness_score(img, others)
        scores.append(score)

    # If new image is more unique than least unique existing image
    new_score = scores[-1]
    min_score = min(scores[:-1])
    if new_score > min_score:
        # Find least unique existing image
        min_index = scores.index(min_score)
        # Remove old image
        os.remove(images[min_index][1])
        # Save new image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(person_folder, f"{timestamp}.jpg")
        cv2.imwrite(filename, new_image)
        return True

    return False


def save_face(face_img, identity):
    path = os.path.join(known_faces_dir, identity)
    if not os.path.exists(path):
        os.makedirs(path)
    return update_person_images(face_img, path)


def find_match(face_img):
    if len(os.listdir(known_faces_dir)) == 0:
        return None

    try:
        for person_dir in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_dir)
            if os.path.isdir(person_path) and os.listdir(person_path):
                reference_img = os.path.join(
                    person_path, os.listdir(person_path)[0])
                result = DeepFace.verify(
                    face_img, reference_img, enforce_detection=False)
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


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
