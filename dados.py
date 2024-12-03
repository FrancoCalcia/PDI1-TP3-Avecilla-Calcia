import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title=None, cmap='gray'):
    """Muestra una imagen con Matplotlib."""
    plt.figure()
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def segmentar_dados_por_color(frame):
    """Segmenta dados rojos traslúcidos en un fondo verdoso."""
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rango de color rojo en HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Crear máscaras para tonos rojos
    mask1 = cv2.inRange(frame_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(frame_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Aplicar la máscara sobre el frame original
    resultado = cv2.bitwise_and(frame, frame, mask=mask)

    return mask, resultado

def detectar_dados_segmentados(frame, umbral=100, min_area=80, max_area=600):
    """Detecta dados usando segmentación de color y detección de contornos."""
    mask, resultado_segmentado = segmentar_dados_por_color(frame)
    imshow(mask, title="Máscara de Dados Rojos")
    
    # Aplicar Canny sobre la máscara
    canny = cv2.Canny(mask, 40, 180)
    bordes_dilatados = cv2.dilate(canny, None, iterations=5)

    contours, _ = cv2.findContours(bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_frame = frame.copy()
    dados = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            dados.append((x, y, w, h))
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    imshow(output_frame, title="Dados Detectados con Segmentación")
    return output_frame, dados

def contar_puntos(frame_bin, dados):
    """Cuenta puntos en cada dado detectado."""
    puntajes = {}
    for i, (x, y, w, h) in enumerate(dados, 1):
        roi = frame_bin[y:y + h, x:x + w]
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, param1=50, param2=15, minRadius=5, maxRadius=15)
        puntajes[f"Dado {i}"] = len(circles[0]) if circles is not None else 0
    return puntajes

def procesar_video(video_path, movement_threshold=30):
    """Procesa un video detectando dados quietos y contando sus puntos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    prev_frame, current_frame, next_frame = None, None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame, current_frame, next_frame = current_frame, next_frame, frame_gray

        if prev_frame is not None and current_frame is not None:
            diff1, diff2 = cv2.absdiff(prev_frame, current_frame), cv2.absdiff(current_frame, next_frame)
            if np.max(diff1) < movement_threshold and np.max(diff2) < movement_threshold:
                print("Dados quietos detectados. Procesando frame...")
                
                frame_dados, dados = detectar_dados_segmentados(frame)
                imshow(frame_dados, title="Dados Detectados con Segmentación", cmap=None)

                _, frame_bin = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 85, 255, cv2.THRESH_BINARY_INV)
                puntajes = contar_puntos(frame_bin, dados)
                print("Resultados de los dados:", puntajes)
                break

    cap.release()

# Procesar video
procesar_video("tirada_4.mp4")
