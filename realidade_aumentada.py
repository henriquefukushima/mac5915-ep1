import cv2
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from pathlib import Path

# ==== parâmetros gerais do código ====
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "aruco-marker.MOV"
MARKER_SIZE = 0.04
TEXTURE_PATH = BASE_DIR / "data" / "wood_texture.png"
VIDEO_ENDED = False

# ==== carregamento dos parâmetros de calibração ====
MATX_PATH = BASE_DIR / "mtx.txt"
DIST_PATH = BASE_DIR / "dist.txt"

mtx = np.loadtxt(MATX_PATH)
dist = np.loadtxt(DIST_PATH)
dist = dist.reshape(1,5)

# ==== dicionário aruco ====
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# ==== VIDEO CAPTURE ====
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Erro ao abrir o vídeo: {VIDEO_PATH}")

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Frame inválido.")

SizeY, SizeX, _ = frame0.shape

# Tamanho original dos frames
orig_h, orig_w = frame0.shape[:2]

# ajustar janela para caber na tela
screen_w = glutGet(GLUT_SCREEN_WIDTH) if glutGet(GLUT_SCREEN_WIDTH) else 1440
screen_h = glutGet(GLUT_SCREEN_HEIGHT) if glutGet(GLUT_SCREEN_HEIGHT) else 900

scale_w = 0.9 * screen_w / orig_w
scale_h = 0.9 * screen_h / orig_h
display_scale = float(min(scale_w, scale_h, 1.0))  # nunca amplia além do original

win_w = int(orig_w * display_scale)
win_h = int(orig_h * display_scale)

# matriz K precisa ser escalonada pela mesma razão do display
K_disp = mtx.copy()
K_disp[0, 0] *= display_scale  # fx
K_disp[1, 1] *= display_scale  # fy
K_disp[0, 2] *= display_scale  # cx
K_disp[1, 2] *= display_scale  # cy

# conversão para matriz de projeção OpenGL
def build_projection_matrix(K, w, h, near=0.001, far=10.0):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    proj = np.zeros((4,4))
    proj[0,0] =  2.0 * fx / w
    proj[1,1] =  2.0 * fy / h
    proj[0,2] =  2.0 * (cx / w) - 1.0
    proj[1,2] =  2.0 * (cy / h) - 1.0
    proj[2,2] = -(far+near)/(far-near)
    proj[3,2] = -1.0
    proj[2,3] = -2.0*far*near/(far-near)
    return proj.T

# conversão pose -> matriz modelview
def build_modelview_matrix(rvec, tvec):
    rvec = np.asarray(rvec).reshape(3, 1)
    tvec = np.asarray(tvec).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    RX = np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # ajuste de convenção de coordenadas
    R = RX @ R
    tvec = RX @ tvec
    modelview = np.eye(4)
    modelview[:3,:3] = R
    modelview[:3,3] = tvec.flatten()
    return modelview.T

# funções de desenho OpenGL
def draw_background(img_bgr, win_w, win_h):
    img = cv2.resize(img_bgr, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OpenGL tem origem no canto inferior; flip vertical necessário
    img = np.flipud(img)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, win_w, 0, win_h, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glRasterPos2i(0, 0)
    glDrawPixels(win_w, win_h, GL_RGB, GL_UNSIGNED_BYTE, img)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_cube(size):
    glBegin(GL_QUADS)
    for sgn in [-1,1]:
        glNormal3f(0,0,sgn)
        for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            glVertex3f(x*size, y*size, sgn*size)
        glNormal3f(sgn,0,0)
        for z,y in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            glVertex3f(sgn*size, y*size, z*size)
        glNormal3f(0,sgn,0)
        for x,z in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            glVertex3f(x*size, sgn*size, z*size)
    glEnd()

def draw_axes(L=0.06):
    glBegin(GL_LINES)
    # X (vermelho)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(L,0,0)
    # Y (verde)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,L,0)
    # Z (azul)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,L)
    glEnd()


# callback OpenGL
current_frame = frame0.copy()
current_rvec, current_tvec = None, None

def show_screen():
    global current_frame, current_rvec, current_tvec

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # desenha background primeiro (2D, sem depth test)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    draw_background(current_frame, win_w, win_h)
    glEnable(GL_DEPTH_TEST)

    if current_rvec is not None:
        # configura projeção com K escalada
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glLoadMatrixd(build_projection_matrix(K_disp, win_w, win_h))

        # configura modelview com pose do marcador
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -MARKER_SIZE/2)
        glLoadMatrixd(build_modelview_matrix(current_rvec, current_tvec))

        glDisable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(1.0, 0.2, 0.2)
        draw_axes(MARKER_SIZE*1.2)
        draw_cube(MARKER_SIZE/4)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # wireframe preto por cima
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0)
        draw_cube(MARKER_SIZE/4)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glutSwapBuffers()


# loop de vídeo
def idle_noop():
    pass

def idle():
    global current_frame, current_rvec, current_tvec, VIDEO_ENDED

    if VIDEO_ENDED:
        return

    if not cap.isOpened():
        VIDEO_ENDED = True
        glutIdleFunc(idle_noop)
        return

    ret, frame = cap.read()
    if not ret:
        print("🏁 Fim do vídeo — mantendo último frame.")
        VIDEO_ENDED = True
        try:
            cap.release()
        except Exception:
            pass
        glutIdleFunc(idle_noop)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    current_rvec, current_tvec = None, None
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = rvec.reshape(3, 1)
            tvec = tvec.reshape(3, 1)
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE*0.5)
            current_rvec, current_tvec = rvec, tvec
            break

    current_frame = frame
    glutPostRedisplay()


def keyboard(key, x, y):
    # fecha com 'q' ou ESC
    if key in (b'q', b'\x1b'):
        try:
            if cap.isOpened():
                cap.release()
        except Exception:
            pass
        try:
            glutHideWindow()
            glutDestroyWindow(glutGetWindow())
        except Exception:
            pass
        import sys
        sys.exit(0)



# configuração OpenGL
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(win_w, win_h)
glutCreateWindow(b"Realidade Aumentada - OpenCV + OpenGL")

glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0,0,1,0])

glutDisplayFunc(show_screen)
glutIdleFunc(idle)
glutKeyboardFunc(keyboard)
glutMainLoop()