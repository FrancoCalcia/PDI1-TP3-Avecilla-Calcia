import cv2
import numpy as np

# Crear la carpeta de salida si no existe
import os
os.makedirs("frames", exist_ok=True)

# Abrir el video
cap = cv2.VideoCapture('tirada_1.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Crear objeto para escribir el video procesado
out = cv2.VideoWriter('Video-Output-Dados.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Parámetros para detección de movimiento
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir frame a escala de grises y aplicar desenfoque
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar movimiento
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Detectar contornos para identificar regiones en movimiento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filtrar áreas pequeñas
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]

        # Procesar ROI para detectar puntos del dado
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
        circles = cv2.HoughCircles(roi_thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=15, minRadius=5, maxRadius=15)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            num_dots = len(circles[0])

            # Dibujar rectángulo y número detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(num_dots), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Actualizar el frame previo
    prev_gray = gray

    # Mostrar el frame procesado
    cv2.imshow('Frame Procesado', frame)

    # Guardar el frame en el video de salida
    out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
