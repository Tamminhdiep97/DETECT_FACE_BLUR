import cv2
import time
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

t = time.time()
name  = 'out'+str(int(t))+'.avi'
# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input 

print("Name video: ",end="")
name_video = input()
cap = cv2.VideoCapture(name_video)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(name, fourcc, int(fps), (frame_width,frame_height))

while(cap.isOpened()):
    # Read the frame
    ret, img = cap.read()
    # Convert to grayscale
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,
            minNeighbors=5,
            minSize=(1,1),
            flags = cv2.CASCADE_SCALE_IMAGE)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            size=min(int(w/10),int(h/10))
            img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],(size,size))
        out.write(img)
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
        #cv2.imshow('img', img)
    else:
        break

# Release the VideoCapture object
out.release()
t2 = time.time()
print("Time:",str(t2-t),"s")
cap.release()