from deepface import DeepFace
import cv2


def process_frame(frame):
    face_results = DeepFace.extract_faces(frame, enforce_detection=False)

    # Draw rectangles around detected faces
    for face_data in face_results:
        if face_data.get('confidence', 0) > 0.5:  # Adjust threshold as needed
            face_coords = face_data['facial_area']
            x = face_coords['x']
            y = face_coords['y']
            w = face_coords['w']
            h = face_coords['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    process_frame(frame)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
