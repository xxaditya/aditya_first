import os
import time
import csv
import pickle
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Global configuration
DATASET_DIR = "dataset"
EMBEDDING_FILE = "output/embeddings.pickle"
EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
PROTO_TXT = "model/deploy.prototxt"
MODEL_FILE = "model/res10_300x300_ssd_iter_140000.caffemodel"
RECOGNIZER_FILE = "output/recognizer.pickle"
LABEL_ENCODER_FILE = "output/le.pickle"
CONFIDENCE_THRESHOLD = 0.5


# Ensure necessary directories exist
def ensure_directories():
    for directory in [DATASET_DIR, "output"]:
        os.makedirs(directory, exist_ok=True)


def create_dataset():
    print("=== Dataset Creation ===")
    # Load face detector
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        print("Error: Unable to load Haar Cascade.")
        return

    # Get user info
    name = input("Enter your name: ").strip()
    roll_number = input("Enter your roll number: ").strip()

    # Setup directory
    user_path = os.path.join(DATASET_DIR, name)
    os.makedirs(user_path, exist_ok=True)

    # Save user info
    with open('student.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, roll_number])

    # Capture images
    cam = cv2.VideoCapture(1)
    total = 0
    while total < 50:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            filepath = os.path.join(user_path, f"{total:05}.png")
            cv2.imwrite(filepath, face)
            total += 1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {name} with {total} images.")


def generate_embeddings():
    print("=== Embedding Generation ===")
    # Load models
    detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL_FILE)
    embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

    imagePaths = list(paths.list_images(DATASET_DIR))
    known_embeddings = []
    known_names = []

    for imagePath in imagePaths:
        print(f"Processing image: {imagePath}")
        name = os.path.basename(os.path.dirname(imagePath))
        image = cv2.imread(imagePath)

        if image is None:
            print(f"Error: Could not load image {imagePath}. Skipping...")
            continue

        # Resize image to larger size for better detection
        image = cv2.resize(image, (500, 500))  # Resize to 500x500 for better detection
        (h, w) = image.shape[:2]

        # Prepare image for face detection
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            # Find the detection with the largest confidence
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Lower the confidence threshold to handle more detections
            if confidence > 0.5:  # Reduced threshold
                print(f"Detection confidence: {confidence}")
                # Calculate bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Clamp bounding box to frame dimensions
                startX = max(0, min(startX, w - 1))
                startY = max(0, min(startY, h - 1))
                endX = max(0, min(endX, w - 1))
                endY = max(0, min(endY, h - 1))

                # Ensure valid bounding box
                if endX > startX and endY > startY:
                    face = image[startY:endY, startX:endX]
                    print(f"Valid face region detected. Dimensions: {face.shape}")

                    # Resize face to match embedding model input
                    face = cv2.resize(face, (96, 96))
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                     (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    known_names.append(name)
                    known_embeddings.append(vec.flatten())
                else:
                    print(f"Invalid bounding box detected: ({startX}, {startY}, {endX}, {endY}). Skipping...")
            else:
                print(f"Low confidence ({confidence}) for detection in {imagePath}. Skipping...")
        else:
            print(f"No faces detected in {imagePath}. Skipping...")

    # Save embeddings
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(EMBEDDING_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    print("Embeddings generated and saved.")




    




def train_model():
    print("=== Training Model ===")
    data = pickle.loads(open(EMBEDDING_FILE, "rb").read())

    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(RECOGNIZER_FILE, "wb") as f:
        f.write(pickle.dumps(recognizer))

    with open(LABEL_ENCODER_FILE, "wb") as f:
        f.write(pickle.dumps(labelEnc))
    print("Model trained and saved.")


def recognize_faces():
    print("=== Face Recognition ===")
    detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL_FILE)
    embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)
    recognizer = pickle.loads(open(RECOGNIZER_FILE, "rb").read())
    le = pickle.loads(open(LABEL_ENCODER_FILE, "rb").read())

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                     (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    text = f"{name}: {proba * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    ensure_directories()
    while True:
        

        print("\n1. Create Dataset")
        print("2. Generate Embeddings")
        print("3. Train Model")
        print("4. Recognize Faces")
        print("5. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            create_dataset()
        elif choice == '2':
            generate_embeddings()
        elif choice == '3':
            train_model()
        elif choice == '4':
            recognize_faces()
        elif choice == '5':
            break
       
