import cv2
import os

# Carpetas de entrada y salida..
input_folder = 'imagenes-umbralar/'
output_folder = 'imagenes-umbraladas/'

# Crea la carpeta de salida si no existe.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Función de umbralado.
def umbralado(imagen, umbral, numero):
    # Convierte la imagen a escala de grises.
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Creación del umbralado.
    umbral, imagen_umbralizada = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)

    # Crea un elemento estructurante de forma cuadrada con tamaño 5x5 píxeles.
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (numero, numero))

    # Aplica la operación morfológica de apertura para eliminar el ruido.
    imagen_procesada = cv2.morphologyEx(imagen_umbralizada, cv2.MORPH_OPEN, se)

    return imagen_procesada

# Definicón de los Datos por default para el umbralado.
umbralado_input = 80
numero = 5

# Lee las imágenes de la carpeta de entrada y procesa cada una.
for filename in os.listdir(input_folder):
    # Lee la imagen.
    imagen = cv2.imread(os.path.join(input_folder, filename))

    i = 0
    while i != 1:
        if i == 2:
            # Obtiene el umbralado deseado.
            umbralado_input = int(input(f"Introduce el valor de umbralado para la imagen: "))
            numero = int(input(f"Introduce el valor del corazon: "))
        
        imagen_umbralizada = umbralado(imagen, umbralado_input, numero)
    
        # Cambia el tamaño de la imagen a 100x100 para mostrarla.
        imagen_umbralizada_pequena = cv2.resize(imagen_umbralizada, (800, 800))
        
        # Muestra la imagen umbralizada en tamaño reducido.
        cv2.imshow('Imagen umbralizada', imagen_umbralizada_pequena)
        cv2.waitKey(600)
        cv2.destroyAllWindows()
        i = int(input(f"Deseas guardar (1-Si o 0-No), ingresa 2-Si quieres modificar parametros: "))

    # Guarda la imagen umbralizada en la carpeta de salida.
    cv2.imwrite(os.path.join(output_folder, filename), imagen_umbralizada)

# Cierra todas las ventanas.
cv2.destroyAllWindows()