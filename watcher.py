import cv2
import os
from deepface import DeepFace
import time
from datetime import datetime

# facenet model?


class FaceWatcher:
    def __init__(self, faces_directory):
        self.faces_directory = faces_directory
        self.detected_people = []
        self.embeddings = {}
        self.load_face_database()

    def load_face_database(self):
        """Load and create embeddings for all faces in the database"""
        print("Loading face database...")

        for person_name in os.listdir(self.faces_directory):
            person_dir = os.path.join(self.faces_directory, person_name)
            if not os.path.isdir(person_dir):
                continue

            # Get image files
            image_files = [f for f in os.listdir(person_dir)
                           if f.lower().endswith(('.png', '.jpg'))]

            if not image_files:
                print(f"No images found for {person_name}")
                continue

            # Load and create embedding for the first image
            image_path = os.path.join(person_dir, image_files[0])
            try:
                embedding = DeepFace.represent(image_path,
                                               model_name="VGG-Face",
                                               enforce_detection=False)
                if embedding:
                    self.embeddings[person_name] = embedding[0]
                    print(f"Loaded reference image for {person_name}")
            except Exception as e:
                print(f"Error processing {person_name}'s image: {str(e)}")

    def find_match(self, face_embedding):
        """Find the closest match for a face embedding in our database"""
        min_distance = float('inf')
        matched_name = None

        for name, ref_embedding in self.embeddings.items():
            # Calculate similarity between embeddings
            distance = DeepFace.verify(face_embedding, ref_embedding,
                                       model_name="VGG-Face",
                                       distance_metric="cosine",
                                       enforce_detection=False)['distance']

            print(f"Distance from {name}: {distance}")

            if distance < min_distance:
                min_distance = distance
                matched_name = name

        # Use a threshold to determine if it's a match
        if min_distance < 0.5:
            return matched_name
        return None

    def process_video(self, video_source=0):
        """Process video stream and detect faces"""
        print("Starting video processing...")
        cap = cv2.VideoCapture(video_source)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Don't process all frames
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            try:
                # Detect and extract faces
                face_results = DeepFace.extract_faces(frame,
                                                      enforce_detection=False)

                for face_data in face_results:
                    # Validate face data
                    if 'facial_area' not in face_data or 'face' not in face_data:
                        print("Invalid face data detected, skipping...")
                        continue

                    face_coords = face_data['facial_area']
                    x = face_coords['x']
                    y = face_coords['y']
                    w = face_coords['w']
                    h = face_coords['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    face = face_data['face']
                    # Validate face image
                    if face is None or face.size == 0:
                        print("Invalid face image, skipping...")
                        continue

                    try:
                        # Ensure face image is in correct format
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        # Get embedding for detected face
                        embedding = DeepFace.represent(face,
                                                       model_name="VGG-Face",
                                                       enforce_detection=False,
                                                       align=True)

                        if embedding and len(embedding) > 0:
                            # Find match in database
                            name = self.find_match(embedding[0])
                            if name and name not in self.detected_people:
                                self.detected_people.append(name)
                                print(f"\nDetected new person: {name}")
                                print(
                                    f"Detected people: {', '.join(self.detected_people)}")
                            else:
                                print("No match found in database")
                    except ValueError as ve:
                        print(f"ValueError processing face: {str(ve)}")
                        continue
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")
                        continue

                # Display the frame
                cv2.imshow('Video', frame)

            except Exception as e:
                print(f"Error processing frame: {str(e)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_system = FaceWatcher("faces_database")
    face_system.process_video()
