# **Detección y Reconocimiento de Caras Visibles en Dados**

## **Descripción del proyecto**

Este proyecto utiliza técnicas avanzadas de procesamiento de imágenes y visión por computadora para detectar dados en reposo y contar las caras visibles a partir de videos. Mediante una combinación de segmentación por color, detección de bordes, análisis de movimiento y Transformada de Hough, se asegura una detección robusta y precisa.

## **Características principales**
- **Detección de dados en videos**: Utiliza segmentación en el espacio de color HSV para identificar dados de colores específicos.
- **Reconocimiento de caras visibles**: Implementa la Transformada de Hough para detectar puntos en las caras de los dados.
- **Reducción de ruido**: Aplica filtros como desenfoque gaussiano y operaciones morfológicas para garantizar contornos claros.
- **Procesamiento optimizado**: Analiza el movimiento en los frames para identificar los dados únicamente cuando están en reposo.

## **Tecnologías utilizadas**
- **OpenCV**: Para procesamiento de imágenes y visión por computadora.
- **Python**: Lenguaje de programación principal del proyecto.
- **Numpy**: Manejo de datos numéricos y matrices.
- **Matplotlib**: Visualizaciones de resultados.

## **Requisitos**
- Python 3.8 o superior.
- Librerías necesarias:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

## **Instrucciones de instalación**
1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/FrancoCalcia/PDI1-TP3-Avecilla-Calcia.git
2. Navega al directorio del proyecto:
   ```bash
   cd PDI1-TP3-Avecilla-Calcia.git
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   
## **Como usar**
1. Ejecuta el script principal:
    ```bash
   python dados.py
2. El script procesará el video, detectará los dados y mostrará las caras visibles en cada frame.
3. El script guardará el video con la detección de los dados en la carpeta `videos`

## **Nota**
Puede que clonar directamente no funcione por el tamaño del repositorio. En caso de ser así, descarga el repositorio como un archivo ZIP desde GitHub:

1. Ve a https://github.com/FrancoCalcia/PDI1-TP3-Avecilla-Calcia en tu navegador.
2. Haz clic en el botón verde `Code` y selecciona `Download ZIP`.
3. Extrae los archivos en tu máquina.
