import cv2

# Open the default camera
video = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Load the Haar cascade classifier for face detection
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Get user ID
id = input("Enter your ID: ")
id = int(id)

# Initialize count for image filenames
count = 0

# Main loop for capturing frames and detecting faces
while True:
    # Read a frame from the video stream
    ret, frame = video.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save images
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite('dataset/User.' + str(id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red color rectangle with thickness 2

    # Display the frame with rectangles
    cv2.imshow("Frame", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Print a message indicating dataset collection is done
print("Dataset collection done")
