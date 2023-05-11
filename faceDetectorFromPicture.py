import cv2

# Load the image
img = cv2.imread('./pictureee.jpg')

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the faces and show the image
for i, (x, y, w, h) in enumerate(faces):
    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Crop the face from the image
    face = img[y:y+h, x:x+w]
    # Save the face as a separate file
    cv2.imwrite(f'face_{i}.jpg', face)

# Show the image with the detected faces
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()