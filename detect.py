import cv2
import numpy as np

video = cv2.VideoCapture("input.mp4")
Har_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    _, frame = video.read()
    cvt_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cvt_scale_1 = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    faces  = Har_cas.detectMultiScale(cvt_scale, 1.1, 5   )
    lower_thres = np.array([30, 38,0])
    higher_thres = np.array([50, 50, 40])
    i=0
    thres = False
    for rect in faces:
        (x,y,w,h) = rect
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h),(204,255,255),2)
        i= i+1
        cv2.putText(frame,"Person"+str(i), (x-12,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    if faces.any() == 0:
        thres = 0
    else: thres = 1
    cv2.putText(frame,"Noise :"+str(thres),(50,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    mask = cv2.inRange(cvt_scale_1,lower_thres,higher_thres)
    cv2.imshow("Frame",frame)
    cv2.imshow("mask",mask)
    print(faces)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.release()
cv2.destroyAllWindows()