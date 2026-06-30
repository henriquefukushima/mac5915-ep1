
import numpy as np
import cv2

#Definindo coordenadas de mundo
#dos pontos de canto do tabuleiro.
objp = np.zeros((9*6, 3), np.float32)
k = 0
for i in range(6):
    for j in range(9):
        objp[k,0] = j  #x
        objp[k,1] = i  #y
        #objp[k,2] = 0 #z
        k += 1
#print(objp)

#Parâmetros do método iterativo para refinamento
#da posição dos cantos nas imagens.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)
Lc = [] #Lista das posições dos cantos nas imagens (c=cantos).
Lw = [] #Lista com as coordenadas de mundo correspondentes (w=world).
for i in range(35):
    img = cv2.imread("dat/cam%02d.jpg"%(i+1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Função do OpenCV para encontrar posições dos cantos do tabuleiro.
    ret,corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        #Método iterativo para refinamento dos cantos.
        corners2 = cv2.cornerSubPix(gray, corners,
                                    (11,11), (-1,-1),
                                    criteria)
        Lc.append(corners2) 
        Lw.append(objp)
        #Função para desenhar os cantos identificados nas imagens.
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imwrite("out/corners%02d.png"%(i+1), img)

h,w = gray.shape
#Função para calibrar a câmera usada de modo a encontrar
#seus parâmetros intrínsecos (mtx e dist) e os parâmetros
#extrínsecos (vetores de rotação rv e translação tv).
ret,mtx,dist,rv,tv = cv2.calibrateCamera(Lw, Lc, (w,h), None, None)
print(mtx)
print(dist)

#Salvando os parâmetros intrínsecos (mtx e dist) para arquivos texto.
np.savetxt("mtx.txt", mtx)
np.savetxt("dist.txt", dist)

#Medindo o erro médio de projeção da calibração em
#relação aos dados observados nas imagens.
mean_error = 0
for i in range(len(Lw)):
    imgp,_ = cv2.projectPoints(Lw[i], rv[i], tv[i], mtx, dist)
    error = cv2.norm(imgp, Lc[i], cv2.NORM_L2)/len(imgp)
    mean_error += error
mean_error = mean_error/len(Lw)
print("Error:",mean_error)

cv2.waitKey(0)
cv2.destroyAllWindows()
