import cv2
import os
from deepface import DeepFace
import time
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceWatcher:
    def __init__(self, faces_directory):
        self.faces_directory = faces_directory
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
            embedding = DeepFace.represent(image_path,
                                           model_name="VGG-Face",
                                           enforce_detection=False)
            if embedding and len(embedding) > 0:
                # Extract the embedding vector from the dictionary
                embedding_vector = embedding[0]['embedding']
                self.embeddings[person_name] = embedding_vector
                print(f"Loaded reference image for {person_name}")

    def find_match(self, face_embedding):
        """Find the closest match for a face embedding in our database"""
        min_distance = float('inf')
        matched_name = None

        # Extract embedding vector from dictionary if needed
        if isinstance(face_embedding, dict) and 'embedding' in face_embedding:
            face_embedding = face_embedding['embedding']
        face_embedding = np.array(face_embedding).reshape(1, -1)

        for name, ref_embedding in self.embeddings.items():
            # Extract embedding vector from dictionary if needed
            if isinstance(ref_embedding, dict) and 'embedding' in ref_embedding:
                ref_embedding = ref_embedding['embedding']
            ref_embedding = np.array(ref_embedding).reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                face_embedding, ref_embedding)[0][0]
            # Convert to distance (1 - similarity)
            distance = 1 - similarity

            # print(f"Distance from {name}: {distance}")

            if distance < min_distance:
                min_distance = distance
                matched_name = name

        # Determine if it's a match
        if min_distance < 0.6:  # Distance threshold
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
            if frame_count % 20 != 0:
                continue

            # Detect and extract faces
            face_results = DeepFace.extract_faces(frame,
                                                  enforce_detection=False)

            for face_data in face_results:
                if face_data.get('confidence', 0) > 0.8:  # Confidence threshold
                    face = face_data['face']
                    # Validate face image
                    if face is None or face.size == 0:
                        print("Invalid face image, skipping...")
                        continue

                    # Convert face image to uint8 if it's not already
                    if face.dtype != np.uint8:
                        face = (face * 255).astype(np.uint8)

                    # Ensure face image is in correct format
                    if len(face.shape) == 3:  # If image is already in color
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    else:  # If image is grayscale
                        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

                    # Get embedding for detected face
                    embedding = DeepFace.represent(face,
                                                   model_name="VGG-Face",
                                                   enforce_detection=False,
                                                   align=True)

                    if embedding and len(embedding) > 0:
                        # Find match in database
                        name = self.find_match(embedding[0])
                        if name:
                            print(f"Match found: {name}")
                        else:
                            print("No match found in database")

                    # Box around face
                    face_coords = face_data['facial_area']
                    x = face_coords['x']
                    y = face_coords['y']
                    w = face_coords['w']
                    h = face_coords['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Display the name if a match is found
                    if name:
                        cv2.putText(frame, name, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_system = FaceWatcher("faces_database")
    face_system.process_video("test_video.mp4")
