#Código mostrado nas aulas que coloca uma caixa com
#textura de madeira sobre o marcador aruco localizado
#em uma imagem fornecida, usando as informações de
#calibração de câmera do arquivo "mtx.txt".
import sys
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
import math

#Limiar de perímetro usado para descartar contornos pequenos.
MINCONTOUR = 200

#Padrões Aruco considerados:
P0 = [[1,0,1,1],
      [0,1,0,1],
      [0,0,1,1],
      [0,0,1,0]]
P1 = [[0,0,0,0],
      [1,1,1,1],
      [1,0,0,1],
      [1,0,1,0]]
P2 = [[0,0,1,1],
      [0,0,1,1],
      [0,0,1,0],
      [1,1,0,1]]
P3 = [[1,0,0,1],
      [1,0,0,1],
      [0,1,0,0],
      [0,1,1,0]]
#Lista dos padrões Aruco:
P = [np.array(P0),
     np.array(P1),
     np.array(P2),
     np.array(P3)]
#Lista da solução encontrada, com os padrões identificados na imagem.
Sol = [None]*4

#Imagem de entrada com padrão aruco.
filename = "./img04.jpg" #"./dat/img04.jpg"  
image = cv2.imread(filename)
if image is None:
    print("Imagem de entrada inválida.")
    sys.exit()

#Redução de imagem para comportar na janela do OpenGL
#(a mesma redução deve ser feita no código de calibração).
image = cv2.resize(image, None, fx=0.2, fy=0.2)
SizeY, SizeX, _ = image.shape
detected = np.copy(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.ADAPTIVE_THRESH_MEAN_C
mask = cv2.adaptiveThreshold(gray, 255,
                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 71, 7)
cv2.imwrite("mask.png", mask)
contours,_ = cv2.findContours(mask,
                              cv2.RETR_LIST, #cv2.RETR_TREE
                              cv2.CHAIN_APPROX_NONE)
#print(len(contours))

#Filtragem de contornos pequenos por limiarização do perímetro:
contours2 = []
for cnt in contours:
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > MINCONTOUR:
        contours2.append(cnt)
#print(len(contours2))

#Aproximação poligonal pelo algoritmo de Douglas-Peucker.
#Filtragem de polígonos sem quatro lados e não convexos.
contours3 = []
for cnt in contours2:
    eps = cv2.arcLength(cnt, True)*0.05
    approx = cv2.approxPolyDP(cnt,eps,True)
    if len(approx) != 4:
        continue
    if cv2.isContourConvex(approx):
        contours3.append(approx)
#print(len(contours3))

#Correção dos vértices dos polígonos para o sentido horário:
for cnt in contours3:
    v01 = [cnt[1,0,0]-cnt[0,0,0], cnt[1,0,1]-cnt[0,0,1]]
    v02 = [cnt[2,0,0]-cnt[0,0,0], cnt[2,0,1]-cnt[0,0,1]]
    pv = np.cross(v01, v02)
    if pv < 0:
        cnt[1,0,0],cnt[3,0,0] = cnt[3,0,0],cnt[1,0,0]
        cnt[1,0,1],cnt[3,0,1] = cnt[3,0,1],cnt[1,0,1]

#Lista de cores dos vértices (vermelho, verde, azul, ciano):
Lc = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
for cnt in contours3:
    pts = []
    for i in range(4):
        x = cnt[i,0,0]
        y = cnt[i,0,1]
        pts.append([x,y])       
    W = H = 300
    #Correção de perspectiva:
    input_pts = np.float32(pts)
    output_pts = np.float32([[  0,   0],
                             [W-1,   0],
                             [W-1, H-1],
                             [  0, H-1]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(gray, M,
                             (W,H),
                             flags=cv2.INTER_LINEAR)
    #Binarização para decodificação do código:
    (T,thresh) = cv2.threshold(out, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Extração do código do marcador:
    dw = W/6
    dh = H/6
    C = []
    y = dh
    for i in range(4):
        x = dw
        linha = []
        for j in range(4):
            x1,y1 = round(x),round(y)
            x2,y2 = round(x+dw),round(y+dh)
            cell = thresh[y1:y2, x1:x2]
            c = cv2.countNonZero(cell)
            if c > dw*dh/2:
                linha.append(1)
            else:
                linha.append(0)
            x += dw
        C.append(linha)
        y += dh
    #print(C)

    #Identificação do marcador:
    #Um teste é necessário para cada rotação do marcador candidato.
    ropt = IDopt = -1
    cod = np.array(C)
    for r in range(4):
        for ID in range(len(P)):
            if (cod == P[ID]).all():
                ropt = r
                IDopt = ID
        #Rotação em 90 graus:
        cod = np.rot90(cod, k=1)
    #print(ropt,IDopt)
    #cv2.imshow("out", thresh)
    #cv2.waitKey(0)
    if IDopt == -1:
        continue
    #Registra a solução identificada com a rotação corrigida:
    input_pts = np.roll(input_pts, shift=-ropt, axis=0)
    Sol[IDopt] = input_pts


#Refinamento da localização dos cantos dos padrões encontrados.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for k in range(len(Sol)):
    if Sol[k] is None:
        continue
    Sol[k] = cv2.cornerSubPix(gray, Sol[k],
                              (11,11), (-1,-1),
                              criteria)

#Gera imagem com as soluções identificadas:
for sol in Sol:
    if sol is None:
        continue
    for i in range(4):
        x,y = int(sol[i,0]),int(sol[i,1])
        cv2.circle(detected,(x,y),16, Lc[i], -1) #16        
#-----------------------
mtx = np.loadtxt("mtx.txt")
dist = np.loadtxt("dist.txt")
dist = dist.reshape(1,5)
#print(dist.shape)

arucoL = 15.4 #cm

p2D = Sol[0]
p3D = np.zeros((4,3), np.float32)
k = 0
for i in range(-1,2,2): #i = -1,1
    for j in range(-1,2,2): #j = -1,1
        p3D[k,0] = j*arucoL/2
        p3D[k,1] = i*arucoL/2
        k += 1
p3D[2,0],p3D[3,0] = p3D[3,0],p3D[2,0]
p3D[2,1],p3D[3,1] = p3D[3,1],p3D[2,1]

success,rv,tv = cv2.solvePnP(p3D,
                             p2D,
                             mtx,
                             dist,
                             flags=0)

#print(success)
print("rv:")
print(rv)
print("tv:")
print(tv)

RX = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

R,_ = cv2.Rodrigues(rv)

Trans = np.zeros((4,4)) 
Trans[:3,:3] = RX@R 
Trans[:3,3:] = RX@tv 
Trans[3,3] = 1.0


#-------- Funções de desenho com OpenGL: ------

#Função para desenhar uma imagem no plano de fundo da janela.
def drawimage(img, px, py):
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    h,w,_ = img.shape
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, w, 0.0, h, 0.0, 200.0)
    glRasterPos2i(px, py) #3f(px, py, pz)
    glDepthMask(GL_FALSE)
    glDrawPixels(w,h,GL_RGB,
                 GL_UNSIGNED_BYTE,
                 np.fliplr(img).tobytes()[::-1])
                 #np.fliplr(img).tostring()[::-1])
    glDepthMask(GL_TRUE)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

#Função para extrair a imagem renderizada na janela em OpenGL.
def getimage(SizeX, SizeY):
    im = glReadPixels(0, 0, SizeX, SizeY,
                      GL_RGB, GL_UNSIGNED_BYTE)
##    t1 = np.copy(np.frombuffer(im, np.uint8))
##    t2 = t1.reshape(SizeY,SizeX,3)
##    ##t3 = t2[::-1, :]  # Read buffer and flip Y
##    t3 = np.flipud(t2)
    t1 = np.copy(np.frombuffer(im, np.uint8)[::-1])
    t2 = t1.reshape(SizeY,SizeX,3)
    t3 = np.fliplr(t2)
    return t3


#Função para desenhar um cubo em OpenGL (em modo imediato).
def drawcube(d, px, py, pz):
    #red =   [1., 0., 0., 1.]
    #glMaterialfv(GL_FRONT,
    #             GL_AMBIENT_AND_DIFFUSE, red)
    C0 = [1.0, 1.0, 1.0, 1.0]
    glMaterialfv(GL_FRONT, GL_SPECULAR, C0)
    glMaterialf(GL_FRONT, GL_SHININESS, 128)
    #----------------
    L = [(-1,-1), (1,-1), (1,1), (-1,1)]
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_LINE)
    for k in range(-1,2,2):
        L.reverse()
        #glColor3f(1.0, 0.0, 1.0)
        C1 = [1.0, 0.0, 1.0, 1.0]
        #glColor4fv(C1)
        glMaterialfv(GL_FRONT,
                     GL_AMBIENT_AND_DIFFUSE, C0) #C1)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, k)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(j*d+px, i*d+py, k*d+pz)
        glEnd()
        #glColor3f(1.0, 1.0, 0.0)
        C2 = [1.0, 1.0, 0.0, 1.0]
        #glColor4fv(C2)
        glMaterialfv(GL_FRONT,
                     GL_AMBIENT_AND_DIFFUSE, C0) #C2)
        glBegin(GL_QUADS)
        glNormal3f(k, 0.0, 0.0)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(k*d+px, j*d+py, i*d+pz)
        glEnd()
        #glColor3f(0.0, 1.0, 1.0)
        C3 = [0.0, 1.0, 1.0, 1.0]
        #glColor4fv(C3)
        glMaterialfv(GL_FRONT,
                     GL_AMBIENT_AND_DIFFUSE, C0) #C3)
        glBegin(GL_QUADS)
        glNormal3f(0.0, k, 0.0)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(i*d+px, k*d+py, j*d+pz)
        glEnd()


#Função para desenhar um elipsoide em OpenGL em modo imediato
#(código explicado em uma das aulas presenciais).
def draw_ellipsoid(a, b, c, slices):
    for i in range(slices): #for(i = 0; i < slices; i++){
        w0 = i / slices
        w1 = (i+1) / slices

        z0 = c*(1.0-w0) + (-c)*w0
        z1 = c*(1.0-w1) + (-c)*w1

        b0 = (b*b*(1.0 - (z0*z0)/(c*c)))**(1/2)
        b1 = (b*b*(1.0 - (z1*z1)/(c*c)))**(1/2)

        a0 = (a*a*(1.0 - (z0*z0)/(c*c)))**(1/2)
        a1 = (a*a*(1.0 - (z1*z1)/(c*c)))**(1/2)

        glBegin(GL_QUAD_STRIP)
        for j in range(0,361,12): #for(j = 0; j <= 360; j++){
            angle = j * (math.pi / 180.0)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            x0 = a0 * cos_angle
            y0 = b0 * sin_angle
            x1 = a1 * cos_angle
            y1 = b1 * sin_angle

            C = [0,0,0]
            C[0] = (2.0*x0)/(a*a)
            C[1] = (2.0*y0)/(b*b)
            C[2] = (2.0*z0)/(c*c)
            mag = (C[0]**2 + C[1]**2 + C[2]**2)**(1/2)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
            glVertex3f( x0, y0, z0 )

            C[0] = (2.0*x1)/(a*a)
            C[1] = (2.0*y1)/(b*b)
            C[2] = (2.0*z1)/(c*c)
            mag = (C[0]**2 + C[1]**2 + C[2]**2)**(1/2)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
            glVertex3f( x1, y1, z1 )
        glEnd()


def draw_ellipsoid_lune(a,b,c, angle1,angle2,slices):
    C = [0.0]*3
    for i in range(0,slices,1):
        w0 = i / slices
        w1 = (i+1) / slices

        z0 = (-c)*(1.0-w0) + c*w0
        z1 = (-c)*(1.0-w1) + c*w1

        b0 = (b*b*(1.0 - (z0*z0)/(c*c)))**(1/2)
        b1 = (b*b*(1.0 - (z1*z1)/(c*c)))**(1/2)

        a0 = (a*a*(1.0 - (z0*z0)/(c*c)))**(1/2)
        a1 = (a*a*(1.0 - (z1*z1)/(c*c)))**(1/2)
    
        glBegin(GL_QUAD_STRIP);
        for j in range(angle1,angle2+1,4):
            angle = float(j) * (math.pi / 180.0)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            x0 = a0 * cos_angle
            y0 = b0 * sin_angle
            x1 = a1 * cos_angle
            y1 = b1 * sin_angle
      
            C[0] = (2.0*x0)/(a*a) #b0*cos_angle
            C[1] = (2.0*y0)/(b*b) #y0
            C[2] = (2.0*z0)/(c*c) #b0*cos_angle;
            mag = (C[0]**2 + C[1]**2 + C[2]**2)**(1/2) #magnitude(C)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
            glVertex3f( x0, y0, z0 )

            C[0] = (2.0*x1)/(a*a) #b1*cos_angle
            C[1] = (2.0*y1)/(b*b) #y1
            C[2] = (2.0*z1)/(c*c) #b1*cos_angle
            mag = (C[0]**2 + C[1]**2 + C[2]**2)**(1/2) #magnitude(C)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
      
            glVertex3f( x1, y1, z1 )
        glEnd()



def draw_curved_cylinder_x(radius0, radius1,
			   bend_radius,
			   bend_ang0,
			   bend_ang1,
			   slices,
			   dtheta):
    C = [0.0]*3
    bend_ang0 *= (math.pi / 180.0)
    bend_ang1 *= (math.pi / 180.0)
    for i in range(0,slices,1):
        w0 = i / slices
        w1 = (i+1) / slices
        ang0 = bend_ang0*(1.0-w0) + bend_ang1*w0
        ang1 = bend_ang0*(1.0-w1) + bend_ang1*w1

        rad0 = radius0*(1.0-w0) + radius1*w0
        rad1 = radius0*(1.0-w1) + radius1*w1

        sin_ang0 = math.sin(ang0)
        cos_ang0 = math.cos(ang0)
        sin_ang1 = math.sin(ang1)
        cos_ang1 = math.cos(ang1)
    
        glBegin(GL_QUAD_STRIP)
        for j in range(0,360+1,dtheta):
            angle = float(j) * (math.pi / 180.0)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            y0 = rad0 * cos_angle + bend_radius
            x0 = rad0 * sin_angle
            y1 = rad1 * cos_angle + bend_radius
            x1 = rad1 * sin_angle
      
            C[0] = x0
            C[1] = sin_ang0*rad0*cos_angle
            C[2] = cos_ang0*rad0*cos_angle
            mag = 0.1+(C[0]**2 + C[1]**2 + C[2]**2)**(1/2) #magnitude(C)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
            yb = sin_ang0 * y0
            xb = x0
            zb = cos_ang0 * y0
            glVertex3f( xb, yb, zb )
      
            C[0] = x1
            C[1] = sin_ang1*rad1*cos_angle
            C[2] = cos_ang1*rad1*cos_angle
            mag = 0.1+(C[0]**2 + C[1]**2 + C[2]**2)**(1/2) #magnitude(C)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
            glNormal3f(C[0], C[1], C[2])
            yb = sin_ang1 * y1
            xb = x1
            zb = cos_ang1 * y1
            glVertex3f( xb, yb, zb )
        glEnd()



def draw_hairy_ellipsoid(a,b,c,slices):
    C = [0.0]*3
    dj = 0
    di = 3
    hair_R = b/3.75 #8.0
    th0 = 90.0
    th1 = 180.0
    c_th0 = math.cos(th0*(math.pi / 180.0))
    s_th0 = math.sin(th0*(math.pi / 180.0))
  
    for i in range(1,int(0.8*slices),1):
        if i%di != 0:
            continue
        w0 = i / slices
        w1 = (i+1) / slices

        z0 = (-c)*(1.0-w0) + c*w0
        z1 = (-c)*(1.0-w1) + c*w1

        b0 = (b*b*(1.0 - (z0*z0)/(c*c)))**(1/2)
        b1 = (b*b*(1.0 - (z1*z1)/(c*c)))**(1/2)

        a0 = (a*a*(1.0 - (z0*z0)/(c*c)))**(1/2)
        a1 = (a*a*(1.0 - (z1*z1)/(c*c)))**(1/2)

        if dj == 0:
            dj = 10
        else:
            dj = 0
    
        for j in range(dj,360+1,20):
            angle = float(j) * (math.pi / 180.0)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            x0 = a0 * cos_angle
            y0 = b0 * sin_angle
            x1 = a1 * cos_angle
            y1 = b1 * sin_angle

            C[0] = (2.0*x0)/(a*a)
            C[1] = (2.0*y0)/(b*b)
            C[2] = (2.0*z0)/(c*c)
            mag = (C[0]**2 + C[1]**2 + C[2]**2)**(1/2) #magnitude(C)
            C[0] /= mag
            C[1] /= mag
            C[2] /= mag
      
            hair_theta = -(math.atan2(x0, -z0)*(180.0/math.pi))/2.0
     
            glPushMatrix()
            glTranslatef(x0 - C[0]*(b/15.0),
		         y0 - C[1]*(b/15.0) - s_th0*hair_R,
		         z0 - C[2]*(b/15.0) - c_th0*hair_R)
            glRotatef(hair_theta, 0.0, 1.0, 0.0 )
            draw_curved_cylinder_x(b/10.0, #3.0,
			           0.0,
			           hair_R,
			           th0,
			           th1,
			           20,
			           12)
            glPopMatrix()



eye_opening = 230 #200
step_eye_opening = 6

def DrawFluffy(d):
    global eye_opening
    global step_eye_opening
    white = [1.0, 1.0, 1.0, 1.0]
    black = [0.0, 0.0, 0.0, 1.0]
    gray  = [0.65, 0.65, 0.65, 1.0]
    red   = [1.0, 0.0, 0.0, 1.0]
    brilho = 128.0
    opaco = 0.0
    #d = 3.0 #30.0
    glDisable(GL_TEXTURE_2D)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
    glMaterialfv(GL_FRONT, GL_SPECULAR, white)
    glMaterialfv(GL_FRONT, GL_SHININESS, 128)
    glPolygonMode(GL_FRONT, GL_FILL)
    a = 0.8*d
    b = d
    c = 0.8*d
    draw_hairy_ellipsoid(a, d, c, 40)

    glPushMatrix()
    glRotatef(-90, 0.0, 1.0, 0.0 )
    draw_ellipsoid_lune(a, b, c, -5, 340, 40)
    glMaterialfv(GL_FRONT, GL_SHININESS, opaco)
    draw_ellipsoid_lune(a*0.95, b*0.95, c*0.95, -16,   0, 40)
    draw_ellipsoid_lune(a*0.95, b*0.95, c*0.95, -40, -18, 40)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, black)
    glMaterialfv(GL_FRONT, GL_SPECULAR, black)
    draw_ellipsoid_lune(a*0.94, b*0.94, c*0.94, -40, 0, 40)
    glMaterialfv(GL_FRONT, GL_SHININESS, brilho)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
    glMaterialfv(GL_FRONT, GL_SPECULAR, white)
    glPopMatrix()
    glPushMatrix()
    glRotatef(90, 1.0, 0.0, 0.0)
    draw_ellipsoid_lune(a, c, b,   0,  60, 40)
    draw_ellipsoid_lune(a, c, b, 120, 180, 40)
    glMaterialfv(GL_FRONT, GL_SHININESS, opaco)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, black)
    glMaterialfv(GL_FRONT, GL_SPECULAR, black)
    draw_ellipsoid_lune(a*0.97, c*0.97, b*0.97,  55,  65, 40)
    draw_ellipsoid_lune(a*0.97, c*0.97, b*0.97, 115, 125, 40)
    glMaterialfv(GL_FRONT, GL_SHININESS, brilho)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
    glMaterialfv(GL_FRONT, GL_SPECULAR, white)
    glPopMatrix()

    #foot:
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, red)
    glPushMatrix()
    glTranslatef( a*0.35, -b,  c*0.2)
    draw_ellipsoid_lune(a*0.30, b*0.10, c*0.70, 0,  180, 40)
    glPopMatrix()
    glPushMatrix()
    glTranslatef( -a*0.35, -b,  c*0.2)
    draw_ellipsoid_lune(a*0.30, b*0.10, c*0.70, 0,  180, 40)
    glPopMatrix()

    #hands:
    glPushMatrix()
    glTranslatef( a*0.95, 0.0,  c*0.75)
    draw_ellipsoid(a*0.15, b*0.30, c*0.40, 40)

    glTranslatef(-a*0.06, 0.0,  c*0.21)
    draw_ellipsoid(a*0.18, b*0.26, c*0.20, 40)
    glTranslatef( a*0.06, 0.0, -c*0.21)

    glTranslatef( -a*0.045, b*0.2, -c*0.1)
    glRotatef(-45.0, 1.0, 0.0, 0.0 )
    glRotatef(-12.0, 0.0, 1.0, 0.0 )
    draw_ellipsoid(a*0.10, b*0.10, c*0.30, 40)
    glPopMatrix()
    glPushMatrix()
    glTranslatef( -a*0.95, 0.0,  c*0.75)
    draw_ellipsoid(a*0.15, b*0.30, c*0.40, 40)

    glTranslatef( a*0.06, 0.0,  c*0.21)
    draw_ellipsoid(a*0.18, b*0.26, c*0.20, 40)
    glTranslatef(-a*0.06, 0.0, -c*0.21)

    glTranslatef( a*0.045, b*0.2, -c*0.1)
    glRotatef(-45.0, 1.0, 0.0, 0.0)
    glRotatef(12.0, 0.0, 1.0, 0.0)  
    draw_ellipsoid(a*0.10, b*0.10, c*0.30, 40)
    glPopMatrix()

    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
  
    glPushMatrix()
    glTranslatef(0.0, b*0.9, c*0.15) #-c*0.5)
    draw_curved_cylinder_x(d/15.0, 0.0,
                           d/3.75,   #32.0,
    			   -20.0, #50.0,
    			   130.0, 20, 12)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-d/6.0, b*0.9, c*0.15)
    glRotatef(30.0, 0.0, 1.0, 0.0)
    draw_curved_cylinder_x(d/15.0, 0.0,
    			   d/3.75, #32.0,
    			 -20.0, #50.0,
    			  130.0, 20, 12)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(d/6.0, b*0.9, c*0.15)
    glRotatef(-30.0, 0.0, 1.0, 0.0)
    draw_curved_cylinder_x(d/15.0, 0.0,
    			   d/3.75, #32.0,
    			 -20.0, #50.0,
    			 130.0, 20, 12)
    glPopMatrix()

    #cauda:
    glPushMatrix()
    glTranslatef(0.0, 0.0, -c*1.05)
    draw_ellipsoid(d*0.30, d*0.30, d*0.30, 10)
    glPopMatrix()
 
    #sobrancelha:
    glPushMatrix()
    glTranslatef(-a*0.15, b*0.55, c*0.75)
    glRotatef(-35.0, 0.0, 0.0, 1.0)
    glRotatef(-30.0, 0.0, 1.0, 0.0)
    draw_ellipsoid(d*0.25, d*0.15, d*0.15, 10)
    glPopMatrix()

    #sobrancelha:
    glPushMatrix()
    glTranslatef(a*0.15, b*0.55, c*0.75)
    glRotatef(35.0, 0.0, 0.0, 1.0 )
    glRotatef(30.0, 0.0, 1.0, 0.0 )
    draw_ellipsoid(d*0.25, d*0.15, d*0.15, 10)
    glPopMatrix()

    #olhos:
    glMaterialfv(GL_FRONT, GL_SHININESS, brilho)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, gray)
    glPushMatrix()
    glTranslatef(0.0, b*0.25, c*0.5)
    glRotatef(90, 0.0, 1.0, 0.0)
    draw_ellipsoid_lune(d*0.45, d*0.45, d*0.45, 270-eye_opening//2, 270+eye_opening//2, 10)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
    draw_ellipsoid_lune(d*0.45, d*0.45, d*0.45, 270+eye_opening//2-360, 270-eye_opening//2, 10)
    glPopMatrix()

    if eye_opening > 340 or eye_opening < 200:
        step_eye_opening = -step_eye_opening
    eye_opening += step_eye_opening

    #focinho:
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, black)
    glPushMatrix()
    glTranslatef(0.0, b*0.15, c*1.05) #0.0, 0.0, c
    draw_ellipsoid(d*0.20, d*0.10, d*0.15, 10)
    glPopMatrix()


#Função para inicialização da matriz de projeção.
def init(SizeX, SizeY):
    near = 0.01 #Near clipping distance
    far  =  100 #Far clipping distance
    glViewport(0, 0, SizeX, SizeY)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #glOrtho(0.0, 500, 0.0, 500, 0.0, 200.0)
    #glFrustum(-SizeX//2, SizeX//2,
    #          -SizeY//2, SizeY//2,
    #          200, 500)
    Proj = np.zeros((4,4))
    Proj[0,0] =  2.0*mtx[0,0]/SizeX
    Proj[1,1] =  2.0*mtx[1,1]/SizeY
    Proj[0,2] = -(2.0*mtx[0,2]/SizeX - 1.0)
    Proj[1,2] =  2.0*mtx[1,2]/SizeY - 1.0
    Proj[2,2] = -(far+near)/(far-near)
    Proj[3,2] = -1.0
    Proj[2,3] = -2.0*far*near/(far-near)
    glLoadMatrixd(np.transpose(Proj).flatten())    
    matrix = glGetDouble(GL_PROJECTION_MATRIX)
    #print("proj matrix:")
    print(np.transpose(matrix))
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


#Função para desenhar o conteúdo da janela.
def showScreen():
    #glLoadIdentity()
    glDepthMask(GL_TRUE)
    glClearColor(1.0, 1.0, 1.0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    drawimage(image, 0, 0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)
    glLoadMatrixd(np.transpose(Trans).flatten())
    glTranslatef(0, 0, -arucoL/6)
    glRotatef(-90.0, 1,0,0)
    glRotatef(180.0, 0,1,0)
    DrawFluffy(arucoL/6)
    glutSwapBuffers()

glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(SizeX, SizeY)
glutInitWindowPosition(0, 0)
wind = glutCreateWindow("OpenGL")

glDepthFunc(GL_LESS)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)
#-------------
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0,0,0,1])
glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3,0.3,0.3,1.])
glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.7,0.7,0.7,1.])
glLightfv(GL_LIGHT0, GL_SPECULAR, [1.,1.,1.,1.])
glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.00002)

#glEnable(GL_LIGHT1)
#glLightfv(GL_LIGHT1, GL_POSITION, [0,200,0,1])
#glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.0,0.0,0.0,1.])
#glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.0,0.0,0.0,1.])
#glLightfv(GL_LIGHT1, GL_SPECULAR, [1.,1.,1.,1.])
#glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.00002)
#glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2,0.2,0.2,1.0])
#-------------
glEnable(GL_TEXTURE_2D)
tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, tex)
img = cv2.imread("wood_texture.png")
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
glTexImage2D(GL_TEXTURE_2D, 0,
             GL_RGB, 512, 512, 0,
             GL_RGB, GL_UNSIGNED_BYTE,
             np.fliplr(img).tobytes()[::-1])
             #np.fliplr(img).tostring()[::-1])

glutDisplayFunc(showScreen)
glutIdleFunc(showScreen)
init(SizeX, SizeY)
showScreen()
#glutMainLoop()

#----------------------------
cv2.imwrite("detected.png", detected)
#cv2.waitKey(2000)
cv2.destroyAllWindows()

imageGL = getimage(SizeX, SizeY)
cv2.imwrite("out.png", imageGL)

glDeleteTextures([tex])
glutHideWindow()
glutDestroyWindow(wind)
print("End")

