import numpy as np
import cv2

image = cv2.imread("./dat2/documento.png")
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

#older version:
#P = cv2.aruco.DetectorParameters_create()
#(corners,ids,rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=P)

#For 4.7.x
P = cv2.aruco.DetectorParameters()
dector = cv2.aruco.ArucoDetector(arucoDict, P)
(corners,ids,rejected) = dector.detectMarkers(image)

Lc = [(0,128,0),(0,255,0),(255,0,0),(255,255,0)]

for i in range(len(corners)):
    Id = ids[i,0]
    Cs = corners[i]
    for j in range(4):
        #x = round(Cs[0,j,0])
        #y = round(Cs[0,j,1])
        C = Cs[0,j]
        x = round(C[0])
        y = round(C[1])
        cv2.circle(image, (x,y), 5, Lc[j], -1)
    x = int(Cs[0,0,0])
    y = int(Cs[0,0,1]) - 15
    cv2.putText(image, str(Id), (x,y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        
cv2.imwrite("detected.png", image)


