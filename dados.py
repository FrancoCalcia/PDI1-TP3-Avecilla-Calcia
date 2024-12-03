import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def imshow(img, title=None, cmap='gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Función para procesar la imagen y detectar dados
def detectar_dados(frame, umbral=100, min_area=80, max_area=600):
    """
    Detecta dados.
    """
    # Aplicar desenfoqe
    desenfoque = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Eliminación de ruido y definición de bordes claros
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_bin = cv2.morphologyEx(desenfoque, cv2.MORPH_OPEN, kernel, iterations=2)
    
    
    # Aplicar Canny para detectar bordes
    canny = cv2.Canny(frame_bin, threshold1=40, threshold2=180)  # Ajusta los umbrales según sea necesario
    bordes_dilatados = cv2.dilate(canny, None, iterations=15)
    
    # Detección de contornos en la imagen de Canny
    contours, _ = cv2.findContours(bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convertir el frame original a BGR para dibujar contornos
    output_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    dados = []

    # Dibujar y filtrar contornos
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expandir ROI para incluir bordes cercanos
            margen_x = int(0.2 * w)
            margen_y = int(0.2 * h)
            x = max(x - margen_x, 0)
            y = max(y - margen_y, 0)
            w = w + 2 * margen_x
            h = h + 2 * margen_y
            
            # Añadir a la lista de dados
            dados.append((x, y, w, h))
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Mostrar imagen con bordes detectados por Canny (opcional)
    imshow(canny, title="Bordes detectados con Canny")

    return output_frame, dados


# Función para contar puntos en los dados
def contar_puntos(frame_bin, dados):
    """
    Cuenta puntos en cada dado detectado.
    """
    puntajes = {}
    puntaje_total = 0

    for i, (x, y, w, h) in enumerate(dados, 1):
        # Recorte del ROI del dado
        roi = frame_bin[y:y + h, x:x + w]

        # Detección de círculos usando HoughCircles
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, param1=50, param2=15, minRadius=5, maxRadius=15)
        puntos = 0

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            puntos = len(circles)

        puntajes[f"Dado {i}"] = puntos
        puntaje_total += puntos

    return puntajes, puntaje_total

# Función principal para procesar un video
# Función principal para procesar un video con detección de movimiento
def procesar_video(video_path, movement_threshold=30):
    """
    Procesa un video para detectar dados y contar sus puntos, asegurándose
    de que estén quietos utilizando comparación de frames consecutivos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Variables para almacenar frames consecutivos
    prev_frame = None
    current_frame = None
    next_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir el frame actual a escala de grises
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Actualizar frames consecutivos
        prev_frame = current_frame
        current_frame = next_frame
        next_frame = frame_gray

        # Comienza a comparar frames solo si hay tres consecutivos
        if prev_frame is not None and current_frame is not None:
            # Calcular diferencias entre los frames
            diff1 = cv2.absdiff(prev_frame, current_frame)
            diff2 = cv2.absdiff(current_frame, next_frame)

            # Comprobar si las diferencias están por debajo del umbral
            if np.max(diff1) < movement_threshold and np.max(diff2) < movement_threshold:
                print("Dados quietos detectados. Procesando frame...")

                # Detectar dados en el frame actual
                frame_dados, dados = detectar_dados(current_frame)

                # Mostrar los dados detectados
                imshow(frame_dados, title="Dados Detectados", cmap=None)

                # Contar puntos en los dados
                _, frame_bin = cv2.threshold(current_frame, thresh=85, maxval=255, type=cv2.THRESH_BINARY_INV)
                puntajes, puntaje_total = contar_puntos(frame_bin, dados)

                # Mostrar resultados
                print("Resultados de los dados:")
                for dado, puntos in puntajes.items():
                    print(f"{dado}: {puntos} puntos")
                print(f"Puntaje total: {puntaje_total}")
                
                break  # Detenemos el procesamiento después de encontrar los dados quietos

    cap.release()


# Ruta del video
video_path = "videos//tirada_1.mp4"

# Procesar el video
procesar_video(video_path)
