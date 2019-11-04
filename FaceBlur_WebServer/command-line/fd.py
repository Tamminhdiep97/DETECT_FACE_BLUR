# load the library using the import keyword
# OpenCV must be properly installed for this to work. If not, then the module will not load with an
# error message.

import cv2
import sys
import random as rd
import time

#get input
print("Name image: ",end="")
link = input()
cascPath = "haarcascade_frontalface_default.xml"

# This creates the cascade classifcation from file 

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# The image is read and converted to grayscale

image = cv2.imread(link)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The face or faces in an image are detected
# This section requires the most adjustments to get accuracy on face being detected.


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Detected {0} faces!".format(len(faces)))

# This draws a green rectangle around the faces detected
image_coppy = image

height = image.shape[0]
width = image.shape[1]
#_blue = img[y,x,0]
#_green = img[y,x,1]
#_red = img[y,x,2]
start_time = time.time()
for (x, y, w, h) in faces:
    sizeKN = rd.randint(25,30)
    i = sizeKN-1
    while i>=0:
        if y-i & y+h+i<height-1:
            if x-w >=0 & x+w+i < width -1:
                image[y-i:y+h+i,x-i:x+w+i] = cv2.blur(image[y-i:y+h+i,x-i:x+w+i],(sizeKN-i,sizeKN-i))
        i=i-1
    #image[y:y+h,x:x+w] = cv2.blur(image[y:y+h,x:x+w],(sizeKN,sizeKN))
    #image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
end_time = time.time()
dur=str((end_time - start_time))
print(dur,"s")

cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
