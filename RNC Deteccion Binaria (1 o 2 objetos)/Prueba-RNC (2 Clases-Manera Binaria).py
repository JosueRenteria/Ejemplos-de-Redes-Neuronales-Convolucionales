# Codigo de pruebas del Modelo.
# Libreria para la extraccion del modelo.
from keras.models import load_model
# Librerias para tratar la imagen.
import cv2
import numpy as np

# Extraemos el Modelo.
classifier = load_model("modelo1_Binario.h5")

# Función para procesar la imagen.
def procesar_imagen(ruta, classifier):
    img = cv2.imread(ruta)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0) # Agrega una dimensión adicional para la cantidad de imágenes

    predictions = classifier.predict(img)

    return predictions

# Función para clasificar.
def clasificar(predictions):
    # Determinamos la prediccion basada en la salida del Modelo.
    if predictions[0][0] >= 0.5:
        prediction = '2'
    else:
        prediction = '1'
    return prediction

# Ruta y Mostrado de los Datos.
print("Esta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba1.jpg', classifier)
print(predictions)
print(clasificar(predictions))

print("\nEsta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba2.jpg', classifier)
print(predictions)
print(clasificar(predictions))

print("\nEsta imagen es un 1")
predictions = procesar_imagen('Imagenes/pruebas/prueba3.jpg', classifier)
print(predictions)
print(clasificar(predictions))

print("\nEsta imagen es un 1")
predictions = procesar_imagen('Imagenes/pruebas/prueba4.jpg', classifier)
print(predictions)
print(clasificar(predictions))

print("\nEsta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba5.jpg', classifier)
print(predictions)
print(clasificar(predictions))