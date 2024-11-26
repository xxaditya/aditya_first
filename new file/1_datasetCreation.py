import imutils
import time
import cv2
import csv
import os

# Path to Haar Cascade XML file
cascade = 'E:/Artificial Intelligence/code17/cascades/haarcascade_frontalface_default.xml'  # Make sure the file is in the correct path

# Initialize the face detector
detector = cv2.CascadeClassifier(cascade)

# Check if the classifier is loaded properly
if detector.empty():
    print("Error: Cascade classifier could not be loaded.")
    exit()

# User input for name and roll number
Name = str(input("Enter your Name: "))
Roll_Number = int(input("Enter your Roll Number: "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

# Create a directory if it doesn't exist
if not os.path.isdir(path):
    os.mkdir(path)
    print(f"Folder created for {sub_data}")

# Save student information to a CSV file
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

print("Starting video stream...")
cam = cv2.VideoCapture(1)  # Check the camera index (it may need to be 0 for your system)
time.sleep(2.0)
total = 0

# Capture 50 frames for dataset
while total < 50:
    print(f"Capturing frame {total + 1}")
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)

    # Detect faces
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    # Loop over the detected faces
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the captured image
        p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1

    # Show the video stream
    cv2.imshow("Frame", frame)

    # Break loop on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release camera and close windows
cam.release()
cv2.destroyAllWindows()
