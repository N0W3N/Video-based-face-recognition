import cv2

cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

stream = cv2.VideoCapture('test.mp4')

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

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()