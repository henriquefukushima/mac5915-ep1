import cv2
import numpy as np
import os
from tqdm import tqdm

os.makedirs("out", exist_ok=True)

MAX_CAPTURAS = 25

# coordenadas de mundo dos pontos de canto do tabuleiro
objp = np.zeros((9*6, 3), np.float32)
k = 0
for i in range(6):
    for j in range(9):
        objp[k,0] = j
        objp[k,1] = i
        k += 1

# parâmetros para refinamento da posição dos cantos
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

Lc = []  # pontos de canto nas imagens
Lw = []  # pontos de mundo correspondentes


cap = cv2.VideoCapture('data/pattern.MOV')
if not cap.isOpened():
    raise RuntimeError("Erro ao abrir o vídeo")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0
capturas_boas = 0
STEP = total_frames // MAX_CAPTURAS

print("\n🎯 Iniciando calibração...\n")

with tqdm(total=total_frames, desc="Processando frames", ncols=80, colour="green") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if frame_id % STEP == 0 and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)

            Lc.append(corners2)
            Lw.append(objp)

            cv2.drawChessboardCorners(frame, (9,6), corners2, found)
            out_path = f"out/corners_{capturas_boas:02d}.png"
            cv2.imwrite(out_path, frame)
            print(f"[ok] quadro {frame_id} -> {out_path}")
            capturas_boas += 1
            
            pbar.set_postfix_str(f"Detecções: {capturas_boas}")

        if capturas_boas >= MAX_CAPTURAS:
            break

        frame_id += 1
        pbar.update(1)

cap.release()
cv2.destroyAllWindows()

print("\n📷 Calibrando a câmera...\n")

h, w = gray.shape
ret, mtx, dist, rv, tv = cv2.calibrateCamera(Lw, Lc, (w,h), None, None)
print("✅ Calibração concluída!")
print(f"→ Capturas boas: {len(Lw)}")
print(f"→ Erro RMS (OpenCV): {ret:.4f}\n")

print("📏 Matriz intrínseca (mtx):\n", mtx)
print("\n🎯 Coeficientes de distorção (dist):\n", dist)

# salva parâmetros de calibração
np.savetxt("mtx.txt", mtx)
np.savetxt("dist.txt", dist)

# mede erro de reprojeção
mean_error = 0
for i in range(len(Lw)):
    imgp, _ = cv2.projectPoints(Lw[i], rv[i], tv[i], mtx, dist)
    error = cv2.norm(imgp, Lc[i], cv2.NORM_L2)/len(imgp)
    mean_error += error
mean_error = mean_error/len(Lw)

print(f"\n📊 Erro médio de reprojeção: {mean_error:.4f}")
print("\n📁 Arquivos salvos: mtx.txt, dist.txt, imagens em ./out/\n")