
import numpy as np
import cv2

#older version:
#arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
#For 4.7.x
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

size = 300
tag = np.zeros((size,size), dtype="uint8")
Id = 0

#older version:
#cv2.aruco.drawMarker(arucoDict, Id, size, tag, 1)
#For 4.7.x
cv2.aruco.generateImageMarker(arucoDict, Id, size, tag, 1)

cv2.imwrite("marker.png", tag)



