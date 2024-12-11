import cv2
from deepface import DeepFace
import os
from datetime import datetime
import numpy as np
import pickle
import hashlib

class FaceManager:
    def __init__(self, known_faces_dir="known_faces", 
                 similarity_threshold=0.5, 
                 max_images_per_person=5):
        self.known_faces_dir = known_faces_dir
        self.similarity_threshold = similarity_threshold
        self.max_images_per_person = max_images_per_person
        
        # Embedding cache to improve performance
        self.embeddings_cache_file = os.path.join(known_faces_dir, "embeddings_cache.pkl")
        self.embeddings_cache = self._load_embeddings_cache()
        
        # Create directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _load_embeddings_cache(self):
        """Load or initialize embeddings cache."""
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def _save_embeddings_cache(self):
        """Save embeddings cache to file."""
        with open(self.embeddings_cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)

    def _generate_image_hash(self, image):
        """Generate a unique hash for an image to use as a cache key."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def _get_face_embedding(self, face_img):
        """Get face embedding, using cache if possible."""
        image_hash = self._generate_image_hash(face_img)
        
        # Check cache first
        if image_hash in self.embeddings_cache:
            return self.embeddings_cache[image_hash]
        
        # Calculate embedding
        try:
            embedding = DeepFace.represent(
                face_img, 
                model_name="VGG-Face", 
                detector_backend="opencv", 
                enforce_detection=False
            )[0]["embedding"]
            
            # Cache the embedding
            self.embeddings_cache[image_hash] = embedding
            self._save_embeddings_cache()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def find_best_match(self, face_img):
        """Find the best match for a face across all known faces."""
        face_embedding = self._get_face_embedding(face_img)
        if face_embedding is None:
            return None
        
        best_match = None
        best_similarity = float('-inf')
        
        for person_dir in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_dir)
            if not os.path.isdir(person_path):
                continue
            
            for img_name in os.listdir(person_path):
                if not img_name.endswith('.jpg'):
                    continue
                
                reference_img_path = os.path.join(person_path, img_name)
                try:
                    result = DeepFace.verify(
                        face_img, 
                        reference_img_path,
                        model_name="VGG-Face",
                        distance_metric="cosine",
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    
                    if result['verified'] and result['distance'] > best_similarity:
                        best_match = person_dir
                        best_similarity = result['distance']
                except Exception:
                    continue
        
        return best_match if best_match and best_similarity > self.similarity_threshold else None

    def get_next_person_id(self):
        """Get the next unique person ID."""
        existing_folders = [
            f for f in os.listdir(self.known_faces_dir)
            if f.startswith("person_") and os.path.isdir(os.path.join(self.known_faces_dir, f))
        ]
        
        if not existing_folders:
            return 0
        
        numbers = [int(f.split('_')[1]) for f in existing_folders]
        return max(numbers) + 1

    def save_face(self, face_img, identity):
        """Save face image, managing the number of images per person."""
        path = os.path.join(self.known_faces_dir, identity)
        os.makedirs(path, exist_ok=True)

        # Get existing images
        existing_images = [
            os.path.join(path, img) for img in os.listdir(path) 
            if img.endswith('.jpg')
        ]

        # If under image limit, save directly
        if len(existing_images) < self.max_images_per_person:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = os.path.join(path, f"{timestamp}.jpg")
            cv2.imwrite(filename, face_img)
            return True

        # More advanced image replacement strategy
        current_embeddings = [self._get_face_embedding(cv2.imread(img)) for img in existing_images]
        new_embedding = self._get_face_embedding(face_img)

        if new_embedding is None:
            return False

        # Find most similar existing image to replace
        similarities = [np.dot(new_embedding, existing_emb) for existing_emb in current_embeddings]
        least_similar_index = similarities.index(min(similarities))

        # Replace least similar image
        os.remove(existing_images[least_similar_index])
        timestamp = datetime.now().strftime("%H%M%S")
        filename = os.path.join(path, f"{timestamp}.jpg")
        cv2.imwrite(filename, face_img)
        return True

    def process_frame(self, frame):
        """Process a single video frame for face detection and tracking."""
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_image, 1.3, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Find or create identity
            identity = self.find_best_match(face_img)
            if identity is None:
                new_id = self.get_next_person_id()
                identity = f"person_{new_id}"

            # Save face image
            self.save_face(face_img, identity)

    def analyze_video(self, video_path):
        """Analyze entire video for face detection and tracking."""
        video_capture = cv2.VideoCapture(video_path)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 == 0:
                print(
                    f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                self.process_frame(frame)

        video_capture.release()

# Usage
if __name__ == "__main__":
    face_manager = FaceManager()
    video_path = "face-demographics-walking-and-pause.mp4"
    face_manager.analyze_video(video_path)