import cv2
import sys

# needed input-files via cli as arguments
filePath = sys.argv[0]
cascPath = sys.argv[1]

# create a stream and cascade file for the process

stream = cv2.VideoCapture(filePath)
faceCascade = cv2.CascadeClassifier(cascPath)

# read stream input and convert each frame into a processable file with the given color space

while stream.isOpened():
    ret, frame = stream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale\
            (
            gray,
            scaleFactor = 1.2, 
            minNeighbors = 5,
            minSize = (30, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
            )
    # draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # show the current frame with a reactangle around the existing faces
        
    cv2.imshow('Frame', frame)
    
    # simple break / close statement 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release() and destroyAllWindows() are needed, otherwise it can result in performance issues on low-spec systems

stream.release()
cv2.destroyAllWindows()
