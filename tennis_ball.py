import cv2
import numpy as np
import math
import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from pathlib import Path

# ==== parâmetros gerais do código ====
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "aruco-marker-3.MOV"
MARKER_SIZE = 0.04
TEXTURE_PATH = BASE_DIR / "data" / "tennis-ball-texture.jpg"
VIDEO_ENDED = False
# variaveis para gravação do video
recording = False
video_writer = None
output_fps = 60.0

# ==== carregamento dos parâmetros de calibração ====
MATX_PATH = BASE_DIR / "mtx.txt"
DIST_PATH = BASE_DIR / "dist.txt"

mtx = np.loadtxt(MATX_PATH)
dist = np.loadtxt(DIST_PATH)
dist = dist.reshape(1,5)

# ==== carregamento da textura ====
def load_texture(texture_path):
    """Load texture from image file"""
    img = cv2.imread(str(texture_path))
    if img is None:
        print(f"Erro ao carregar textura: {texture_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Generate texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    
    return texture_id

# Don't load texture here - will load after OpenGL context is created
tennis_texture = None

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

# ==== ajustar janela para caber na tela ====
screen_w = glutGet(GLUT_SCREEN_WIDTH) if glutGet(GLUT_SCREEN_WIDTH) else 1440
screen_h = glutGet(GLUT_SCREEN_HEIGHT) if glutGet(GLUT_SCREEN_HEIGHT) else 900

scale_w = 0.9 * screen_w / orig_w
scale_h = 0.9 * screen_h / orig_h
display_scale = float(min(scale_w, scale_h, 1.0))  # nunca amplia além do original

win_w = int(orig_w * display_scale)
win_h = int(orig_h * display_scale)

# K (intrínseca) precisa ser escalonada pela MESMA razão de escala do display
K_disp = mtx.copy()
K_disp[0, 0] *= display_scale  # fx
K_disp[1, 1] *= display_scale  # fy
K_disp[0, 2] *= display_scale  # cx
K_disp[1, 2] *= display_scale  # cy

# ======= CONVERSÃO PARA MATRIZ DE PROJEÇÃO OPENGL =======
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

# ======= CONVERSÃO POSE -> MATRIZ MODELVIEW =======
def build_modelview_matrix(rvec, tvec):
    rvec = np.asarray(rvec).reshape(3, 1)   # <— garante (3,1)
    tvec = np.asarray(tvec).reshape(3, 1)   # <— garante (3,1)
    R, _ = cv2.Rodrigues(rvec)
    RX = np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # ajuste de convenção
    R = RX @ R
    tvec = RX @ tvec
    modelview = np.eye(4)
    modelview[:3,:3] = R
    modelview[:3,3] = tvec.flatten()
    return modelview.T

# ======= FUNÇÕES DE DESENHO OPENGL =======
def draw_background(img_bgr, win_w, win_h):
    # Redimensiona ao tamanho da janela
    img = cv2.resize(img_bgr, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OpenGL tem origem no canto inferior; fazemos flip vertical para ficar "em pé"
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

def draw_sphere(radius, slices=20, stacks=20):
    """Draw a sphere with given radius, slices (longitude), and stacks (latitude)"""
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            # Normal for lighting
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

def draw_tennis_ball_seam(radius, num_points=200):
    """Draw a single continuous tennis ball seam that wraps around the sphere"""
    glColor3f(1.0, 1.0, 1.0)  # White lines
    glLineWidth(4.0)
    
    glBegin(GL_LINE_STRIP)
    for i in range(num_points + 1):
        t = 2 * math.pi * float(i) / num_points
        
        # Tennis ball seam formula - creates a single continuous curve
        # This creates a figure-8 pattern that wraps around the sphere
        x = (3 * math.sin(t) + math.sin(3 * t)) / 4
        y = (3 * math.cos(t) - math.cos(3 * t)) / 4
        z = (math.sqrt(3) * math.cos(2 * t)) / 2
        
        # Normalize to unit sphere, then scale by radius
        length = math.sqrt(x*x + y*y + z*z)
        if length > 0:
            x = (x / length) * radius
            y = (y / length) * radius
            z = (z / length) * radius
        
        glVertex3f(x, y, z)
    glEnd()

def draw_tennis_ball(radius, slices=20, stacks=20):
    """Draw a tennis ball with texture mapping and characteristic curved lines"""
    # Enable texturing
    if tennis_texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tennis_texture)
        glColor3f(1.0, 1.0, 1.0)  # White to show texture colors
    else:
        glDisable(GL_TEXTURE_2D)
        glColor3f(0.0, 0.6, 0.0)  # Fallback green color
    
    # Draw textured sphere
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            # Calculate texture coordinates
            u0 = float(j) / slices
            v0 = float(i) / stacks
            u1 = float(j) / slices
            v1 = float(i + 1) / stacks
            
            # Normal for lighting
            glNormal3f(x * zr0, y * zr0, z0)
            glTexCoord2f(u0, v0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            
            glNormal3f(x * zr1, y * zr1, z1)
            glTexCoord2f(u1, v1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()
    
    # Disable texturing for the lines
    glDisable(GL_TEXTURE_2D)
    
    # Enable polygon offset to draw lines slightly in front of the sphere
    glEnable(GL_POLYGON_OFFSET_LINE)
    glPolygonOffset(-1.0, -1.0)
    
    # Draw the single continuous tennis ball seam
    draw_tennis_ball_seam(radius)
    
    # Disable polygon offset
    glDisable(GL_POLYGON_OFFSET_LINE)

def draw_axes(L=0.06):
    glBegin(GL_LINES)
    # X (vermelho)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(L,0,0)
    # Y (verde)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,L,0)
    # Z (azul)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,L)
    glEnd()

# ======= CALLBACK OPENGL =======
current_frame = frame0.copy()
current_rvec, current_tvec = None, None

def show_screen():
    global current_frame, current_rvec, current_tvec

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw background first (2D, no depth testing)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    draw_background(current_frame, win_w, win_h)
    glEnable(GL_DEPTH_TEST)

    if current_rvec is not None:
        # PROJEÇÃO (reset + carrega sua K escalada)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glLoadMatrixd(build_projection_matrix(K_disp, win_w, win_h))

        # MODELVIEW (reset + carrega a pose do marcador)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -MARKER_SIZE/4)
        glLoadMatrixd(build_modelview_matrix(current_rvec, current_tvec))

        # Leve afastamento do plano do marcador — tente + (para "cima")
        # Position the ball floating above the marker
        glTranslatef(0.0, 0.0, MARKER_SIZE * 1)  # Move up by MARKER_SIZE units

        # Preenchido (sem iluminação) para garantir cor visível
        glDisable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(1.0, 0.2, 0.2)
        #draw_axes(MARKER_SIZE*1.2)
        draw_tennis_ball(MARKER_SIZE/3)  # Tennis ball instead of cube
        glDisable(GL_POLYGON_OFFSET_FILL)

    glutSwapBuffers()
    save_frame_to_video()


# ======= LOOP DE VÍDEO =======
def idle_noop():
    pass

def idle():
    global current_frame, current_rvec, current_tvec, VIDEO_ENDED

    if VIDEO_ENDED:
        return

    if not cap.isOpened():
        
        VIDEO_ENDED = True
        glutIdleFunc(idle_noop)  # desliga o idle

        return

    ret, frame = cap.read()
    if not ret:
        print("🏁 Fim do vídeo — mantendo último frame.")
        VIDEO_ENDED = True
        try:
            cap.release()
        except Exception:
            pass
        glutIdleFunc(idle_noop)  # <- chave pra parar o loop
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    current_rvec, current_tvec = None, None
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = rvec.reshape(3, 1)   
            tvec = tvec.reshape(3, 1)  
            #cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE*0.5)
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
    elif key == b's':  # Press 's' to save current frame
        save_current_frame()
    elif key == b'r':  # Press 'r' to start/stop recording
        toggle_recording()

def toggle_recording():
    """Start or stop video recording"""
    global recording, video_writer
    if not recording:
        # Start recording
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f"tennis_ball_output_{int(time.time())}.mp4"
        video_writer = cv2.VideoWriter(filename, fourcc, output_fps, (win_w, win_h))
        recording = True
        print(f"Started recording: {filename}")
    else:
        # Stop recording
        if video_writer:
            video_writer.release()
            video_writer = None
        recording = False
        print("Stopped recording")

def save_current_frame():
    """Save the current OpenGL frame as an image"""
    # Read pixels from OpenGL framebuffer
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, win_w, win_h, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Convert to numpy array and flip vertically
    img = np.frombuffer(pixels, dtype=np.uint8)
    img = img.reshape((win_h, win_w, 3))
    img = np.flipud(img)  # OpenGL has origin at bottom-left
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Save image
    filename = f"tennis_ball_opengl_{int(time.time())}.jpg"
    cv2.imwrite(filename, img)
    print(f"OpenGL frame saved as {filename}")

def save_frame_to_video():
    """Save current OpenGL frame to video if recording"""
    global video_writer, recording
    if recording and video_writer is not None:
        # Read pixels from OpenGL framebuffer
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, win_w, win_h, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and flip vertically
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape((win_h, win_w, 3))
        img = np.flipud(img)  # OpenGL has origin at bottom-left
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write to video
        video_writer.write(img)

# ======= CONFIGURAÇÃO OPENGL =======
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(win_w, win_h)
glutCreateWindow(b"Tennis Ball AR - OpenCV + OpenGL")

glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0,0,1,0])

# Enable texturing
glEnable(GL_TEXTURE_2D)

# Load texture after OpenGL context is created
tennis_texture = load_texture(TEXTURE_PATH)
if tennis_texture is None:
    print("Warning: Could not load tennis ball texture, using fallback color")

glutDisplayFunc(show_screen)
glutIdleFunc(idle)
glutKeyboardFunc(keyboard)
glutMainLoop()
