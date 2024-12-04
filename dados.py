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

    return mask

def detectar_contornos(frame):
    """Detecta los contornos de los dados en un frame dado."""
    mask = segmentar_dados_por_color(frame)
    
    # Aplicamos Canny para detectar bordes y encontrar los contornos
    canny = cv2.Canny(mask, 1000, 1500)
    bordes_dilatados = cv2.dilate(canny, None, iterations=2)

    # Detectamos los contornos
    contours, _ = cv2.findContours(bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def comparar_contornos(contours_prev, contours_current, umbral_area=45):
    """Compara los contornos entre dos frames basándose en el área total."""
    areas_prev = [cv2.contourArea(contour) for contour in contours_prev]
    areas_current = [cv2.contourArea(contour) for contour in contours_current]
    
    area_prev_total = sum(areas_prev)
    area_current_total = sum(areas_current)

    # Si la diferencia en el área total es menor que un umbral, consideramos que los dados están quietos
    if abs(area_prev_total - area_current_total) < umbral_area:
        return True  # Los dados están quietos
    return False  # Los dados están en movimiento

def contar_caras_dado(dado_recortado):
    """
    Cuenta las caras visibles de un dado en un recorte dado.
    1. Convierte a escala de grises.
    2. Aplica Canny para detectar bordes.
    3. Utiliza la Transformada de Hough para detectar círculos.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(dado_recortado, cv2.COLOR_BGR2GRAY)
    #imshow(gray)
   
    # Aplicar desenfoque para mejorar la detección de bordes
    blurred = cv2.GaussianBlur(gray, (9, 9), 3)
    #imshow(blurred)
   
    # Aplicar Canny para detectar bordes
    edges = cv2.Canny(blurred, 30, 150)
    imshow(edges)
    # Aplicar la Transformada de Hough para detectar círculos
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,         # Resolución inversa del acumulador
        minDist=8,     # Distancia mínima entre los centros de los círculos
        param1=25,      # Umbral superior para Canny
        param2=10,      # Umbral del acumulador para considerar un círculo válido
        minRadius=5,   # Radio mínimo esperado
        maxRadius=8    # Radio máximo esperado
    )

    # Contar los círculos detectados
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dibujar los círculos detectados (opcional)
            cv2.circle(dado_recortado, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Mostrar la imagen con círculos detectados
        imshow(dado_recortado, title=f"{len(circles[0])} caras detectadas")
        return len(circles[0])
    
    # Si no se detectan círculos, el dado tiene 0 caras visibles
    return 0

def recortar_y_contar_dados(frame, contours, min_area=4300, max_area=6400):
    """
    Recorta los dados basados en contornos dentro de un rango de área
    y cuenta las caras visibles de cada dado.
    """
    resultados = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            dado_recortado = frame[y:y+h, x:x+w]
            
            # Contar caras visibles
            caras = contar_caras_dado(dado_recortado)
            resultados.append((area, caras))

            print(f"Dado con área {area:.2f}: {caras} caras detectadas")
    return resultados


def procesar_video_para_quietud(video_path, tiempo_espera=1.58):
    """Procesa un video y detecta el segundo exacto en que los dados se quedan quietos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Obtener el tiempo actual en segundos
        current_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        
        # Ignorar los primeros segundos
        if current_second < tiempo_espera:
            prev_frame = frame
            continue
        
        if prev_frame is not None:
            # Detectamos los contornos en el frame anterior y el actual
            contours_prev = detectar_contornos(prev_frame)
            contours_current = detectar_contornos(frame)
            
            # Comparamos los contornos para verificar si los dados están quietos
            if comparar_contornos(contours_prev, contours_current):
                print(f"Dados quietos detectados en el segundo: {current_second:.2f}")
                
                # Mostrar el frame donde se detecta la quietud
                #imshow(frame, title=f"Dados quietos en el segundo {current_second:.2f}")

                # Dibujar los contornos en el frame
                frame_contornos = frame.copy()
                cv2.drawContours(frame_contornos, contours_current, -1, (0, 255, 0), 2)
                #imshow(frame_contornos, title="Contornos detectados")

                # Recortar los dados y contar caras
                resultados = recortar_y_contar_dados(frame, contours_current)
                print("Resultados finales:", resultados)

                break  # Salimos después de detectar la quietud

        prev_frame = frame

    cap.release()


tiradas = [1, 2, 3, 4]
for tirada in tiradas:
    procesar_video_para_quietud(f"videos/tirada_{tirada}.mp4")

