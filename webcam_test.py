from deepface import DeepFace
import cv2


def process_frame(frame):
    try:
        face_results = DeepFace.extract_faces(frame, enforce_detection=False)

        # Draw rectangles around detected faces
        for face_data in face_results:
            face_coords = face_data['facial_area']
            x = face_coords['x']
            y = face_coords['y']
            w = face_coords['w']
            h = face_coords['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error processing frame: {str(e)}")


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
