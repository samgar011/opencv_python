import cv2
import numpy as np

cap = cv2.VideoCapture(0)
mog2 = cv2.createBackgroundSubtractorMOG2()
while True:
    ret,frame = cap.read() # 30 fps 
    buyumeFaktor = 0.3
    interpolation = cv2.INTER_AREA
    frame = cv2.resize(frame,None,fx=buyumeFaktor,fy=buyumeFaktor,interpolation=interpolation)
    maske = mog2.apply(frame)
    cv2.imshow("resim",frame)
    cv2.imshow("maske",maske)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()