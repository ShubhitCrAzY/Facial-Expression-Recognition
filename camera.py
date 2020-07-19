import cv2
from model import FacialExpressionModel
import numpy as np

facec=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#This is an inbuilt opencv classifier for face detection
model=FacialExpressionModel('model.json','model_weights.h5')
#This is the model required for prediction
font=cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, fr = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        gray_fr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        faces=facec.detectMultiScale(gray_fr,1.3,5)

        for (x,y,w,h) in faces:
            fc=gray_fr[y:y+h, x:x+w]

            roi=cv2.resize(fc,(48,48))
            pred=model.predict_emotion(roi[np.newaxis,:,:,np.newaxis])

            cv2.putText(fr,pred,(x,y),font,1,(255,255,0),2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)



        ret, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
