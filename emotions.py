# Import Modules
import cv2

# Classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')  

# Grab Webcam feed
webcam = cv2.VideoCapture(0)

# Check if the webcam started successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Show the current frame
while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # If there's an error, abort mission
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)

    # Run the face detection
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 3)

        # Get the subframe (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale for the_face
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Detect eyes
        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.1, minNeighbors=10)

        # Label the face based on the detection results
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=2)
        if len(eyes) >= 2:
            cv2.putText(frame, 'Awake', (x, y+h+70), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
        elif len(eyes) < 2:
            cv2.putText(frame, 'Sleepy', (x, y+h+70), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)

    # Show the current webcam frame
    cv2.imshow('The Emotion Detector', frame)

    # Display the frame
    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

Final = "Mission Accomplished! LOL..."
print(Final)
