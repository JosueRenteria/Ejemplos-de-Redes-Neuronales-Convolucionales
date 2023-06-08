import cv2
import numpy as np

# Ingreso y procesamiento de la imagen.
imagen = cv2.imread('imagenes/4-2.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
umbral, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

# Se Crea la dilatación.
kernel = np.ones((10, 10), np.uint8)
dilatacion = cv2.dilate(binaria, kernel, iterations=3)
contornos, jerarquia = cv2.findContours(dilatacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ver y procesar los contornos de las figuras.
mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)
for contorno in contornos:
    cv2.drawContours(mascara, [contorno], 0, 255, -1)

# Intentar obtener los objetos de todas las imagenes.
objetos_separados = cv2.bitwise_and(imagen, imagen, mask=mascara)
contornos_separados, jerarquia_separados = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Resultados y numero de objetos obtenidos.
cantidad_objetos = len(contornos_separados)
print("Cantidad de objetos encontrados:", cantidad_objetos)