import numpy as np
import imutils
import pickle
import time
import cv2
import os

# Paths to model files and other resources
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "embeddings.pickle"
recognizerFile = "recognizer.pickle"
labelEncFile = "le.pickle"
conf = 0.5

# Check if required files exist
if not os.path.exists(embeddingModel):
    print("Error: PyTorch embedding model '{}' not found.".format(embeddingModel))
    exit()

if not os.path.exists(recognizerFile):
    print("Error: Recognizer model '{}' not found.".format(recognizerFile))
    exit()

if not os.path.exists(labelEncFile):
    print("Error: Label encoder '{}' not found.".format(labelEncFile))
    exit()

# Loading the face detection model
print("Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
if not os.path.exists(prototxt) or not os.path.exists(model):
    print("Error: Caffe model or prototxt file not found.")
    exit()

detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Loading the face recognition models
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

print("Starting video stream...")
cam = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create a blob for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Extract face embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict face label
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}  : {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # Draw bounding box and text
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Display the video stream
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to quit
        break

cam.release()
cv2.destroyAllWindows()
