import cv2
import numpy as np
#
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

capture = cv2.VideoCapture(0)

while 1:
    ret, img = capture.read()
    #img = cv2.imread('Aaron_Eckhart_0001.jpg')
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayscale, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_colour = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        nose = noseCascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_colour, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
        mouth = mouthCascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_colour, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

    #text = "red - mouth \n green - eyes \n blue - face \n yellow - nose"
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 10, 40
    font_size = 0.5
    font_thickness = 1
    wrapped = ['Colour: ', 'Blue - Face', 'Green - Eyes', 'Yellow - Nose', 'Red - Mouth', ' ','Faces Count : ' + str(len(faces))]
    i = 0
    for line in wrapped:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 5

        y = int((img.shape[0] + textsize[1]) / 2) + i * gap
        x = 10

        cv2.putText(img, line, (x, y), font,
                    font_size,
                    (0, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA)
        i += 1


    cv2.imshow('Detect Face', img)
    k = cv2.waitKey(30) & 0xff
    if k == 13:
        break
capture.release()
cv2.destroyAllWindows()