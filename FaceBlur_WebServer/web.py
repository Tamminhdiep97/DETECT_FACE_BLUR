from flask import Flask, render_template, request, send_from_directory, redirect
import glob, os
from werkzeug.utils import secure_filename
import numpy as np
import argparse
import time
import cv2
import os
import re
import threading

from flask import flash

import sys
import random as rd
import time

#Blur_Img function
def blur_img(Img1):
    gray = cv2.cvtColor(Img1, cv2.COLOR_BGR2GRAY)
    t1=time.time()              #time start

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1,1),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Detected {0} faces!".format(len(faces)))

# This draws a green rectangle around the faces detected
    image_coppy = Img1

    height = Img1.shape[0]
    width = Img1.shape[1]
    start_time = time.time()

    for (x, y, w, h) in faces:
        sizeKN = rd.randint(25,30)
        i = sizeKN-1
        while i>=0:
            if y-i & y+h+i<height-1:
                if x-w >=0 & x+w+i < width -1:
                    Img1[y-i:y+h+i,x-i:x+w+i] = cv2.blur(Img1[y-i:y+h+i,x-i:x+w+i],(sizeKN-i,sizeKN-i))
            i=i-1
        Img1[y:y+h,x:x+w] = cv2.blur(Img1[y:y+h,x:x+w],(sizeKN,sizeKN))
        #Img1 = cv2.rectangle(Img1,(x,y),(x+w,y+h),(255,0,0),1)
    end_time = time.time()
    #t2=str(int(end_time - start_time))
    t3 = int(end_time)

    cv2.imwrite(os.path.sep.join(["static/images/", "result"+str(t3)+".jpg"]),Img1)
    return t3

def blur_video(cap):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    t3 = int(time.time())
    out = cv2.VideoWriter(os.path.sep.join(["static/videos", "result"+str(t3)+".mp4"]),fourcc, int(fps), (frame_width,frame_height))

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
                sizeKN = min(int(w/10),int(h/10))                
                img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],(sizeKN,sizeKN))
                
            #img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],(27,27))
            out.write(img)
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
        #cv2.imshow('img', img)
        else:
            break
    # Stop if escape key is pressed
# Release the VideoCapture object
    out.release()

    cap.release()
    return t3


#webFunction
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpg','mp4','avi'])


DEBUG = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/image-search", methods=['POST'])

def image_search():
    #Get data from form
    if request.method == 'POST':
        if 'file_original' not in request.files:
            flash('Imgs or Videos type wrong')
            return redirect(request.url)

        file1 = request.files['file_original']
        if file1.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file1 and allowed_file(file1.filename):
            filename1 = secure_filename(file1.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))

               
        #End: Get data from form
        path=[]
        files=[]
        #Start: Blur face in Img
        if filename1.find(".jpg") != -1:
            image = cv2.imread(os.path.sep.join(["static/uploads/", filename1])) #load img_Original           
            t3 = blur_img(image)
            path.append(os.path.sep.join(["static/images", "result"+str(t3)+".jpg" ]))
            files.append(glob.glob(path[0]))
            return render_template('image-search.html', files=files, path=path)
        #End: Blur face in Img

        if filename1.find(".mp4") != -1 or filename1.find(".avi") != -1:
            print("COME HERE")
            Cap = cv2.VideoCapture(os.path.sep.join(["static/uploads/", filename1]))
            t3 = blur_video(Cap)
            path.append(os.path.sep.join(["static/videos", "result"+str(t3)+".mp4" ]))
            files.append(glob.glob(path[0]))
            return render_template('videos.html', files=files, path=path)

#print(result2d)
       


        
        
        #files.append(glob.glob(path[0]))
        #return render_template('image-search.html', files=files, path=path)
    


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images/', path)

@app.route('/images/<path:path>')
def send_video(path):
    return send_from_directory('videos/',path)
if __name__ == "__main__":
	app.run()
