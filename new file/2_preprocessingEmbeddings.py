from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Directories and model files
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
prototxt = "model/deploy.prototxt"
model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

# Ensure the necessary directories exist
os.makedirs('output', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load models
if not os.path.exists(prototxt) or not os.path.exists(model):
    print("Error: Caffe model or prototxt file not found.")
    exit()
if not os.path.exists(embeddingModel):
    print("Error: PyTorch embedding model not found.")
    exit()

detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialize lists for storing embeddings and names
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Process images and extract embeddings
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # Convert image to blob for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    # Detect faces
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            # Extract face region of interest (ROI)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Create face blob and extract embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print(f"Embeddings: {total}")
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Save embeddings and names to pickle file
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Process Completed")
