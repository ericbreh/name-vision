import cv2
import os
from deepface import DeepFace
import time
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, faces_directory):
        self.faces_directory = faces_directory
        self.detected_people = []
        self.reference_embeddings = {}
        self.load_face_database()
        
    def load_face_database(self):
        """Load and create embeddings for all faces in the database"""
        print("Loading face database...")
        
        for person_name in os.listdir(self.faces_directory):
            person_dir = os.path.join(self.faces_directory, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            # Get first image file for each person
            image_files = [f for f in os.listdir(person_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
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
                    self.reference_embeddings[person_name] = embedding[0]
                    print(f"Loaded reference image for {person_name}")
            except Exception as e:
                print(f"Error processing {person_name}'s image: {str(e)}")

    def find_match(self, face_embedding):
        """Find the closest match for a face embedding in our database"""
        min_distance = float('inf')
        matched_name = None
        
        for name, ref_embedding in self.reference_embeddings.items():
            # Calculate cosine similarity between embeddings
            distance = DeepFace.verify(face_embedding, ref_embedding,
                                     model_name="VGG-Face",
                                     distance_metric="cosine",
                                     enforce_detection=False)['distance']
            
            if distance < min_distance:
                min_distance = distance
                matched_name = name
        
        # Use a threshold to determine if it's a match
        if min_distance < 0.3:  # You may need to adjust this threshold
            return matched_name
        return None

    def process_video(self, video_source=0):
        """Process video stream and detect faces"""
        print("Starting video processing...")
        cap = cv2.VideoCapture(video_source)
        
        # Set a lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 3rd frame to improve performance
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            try:
                # Detect and extract faces
                face_results = DeepFace.extract_faces(frame, 
                                                    enforce_detection=False)
                
                for face_data in face_results:
                    face = face_data['face']
                    try:
                        # Get embedding for detected face
                        embedding = DeepFace.represent(face, 
                                                     model_name="VGG-Face",
                                                     enforce_detection=False)
                        if embedding:
                            # Find match in database
                            name = self.find_match(embedding[0])
                            if name and name not in self.detected_people:
                                self.detected_people.append(name)
                                print(f"\nDetected new person: {name}")
                                print(f"Current people in scene: {', '.join(self.detected_people)}")
                    except Exception as e:
                        continue  # Skip this face if there's an error

                # Display the frame
                cv2.imshow('Video', frame)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    face_system = FaceRecognitionSystem("faces_database")
    face_system.process_video()  # Use 0 for webcam or provide video file path