# Codigo de pruebas del Modelo.
from keras.preprocessing.image import ImageDataGenerator
# Libreria para la extraccion del modelo.
from keras.models import load_model
# Librerias para tratar la imagen.
import cv2
import numpy as np

# Extraemos el Modelo.
classifier = load_model("modelo2_3Clases (Buen Modelo).h5")

# Función para procesar la imagen.
def procesar_imagen(ruta, classifier):
    img = cv2.imread(ruta)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0) # Agrega una dimensión adicional para la cantidad de imágenes

    predictions = classifier.predict(img)

    return predictions

# Función para clasificar.
def clasificar(predictions):
    # Elementos que hay.
    lista = ['1', '2', '3']

    # Obtener el índice de la categoría con la probabilidad más alta
    cat_index = np.argmax(predictions)
    elementos = lista[cat_index]

    # Retornamos las predicciones.
    return elementos

# Ruta y Mostrado de los Datos.
print("1_Esta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba1.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n2_Esta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba2.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n3_Esta imagen es un 1")
predictions = procesar_imagen('Imagenes/pruebas/prueba3.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n4_Esta imagen es un 1")
predictions = procesar_imagen('Imagenes/pruebas/prueba4.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n5_Esta imagen es un 2")
predictions = procesar_imagen('Imagenes/pruebas/prueba5.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n6_Esta imagen es un 3")
predictions = procesar_imagen('Imagenes/pruebas/prueba6.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n7_Esta imagen es un 3")
predictions = procesar_imagen('Imagenes/pruebas/prueba7.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n8_Esta imagen es un 3")
predictions = procesar_imagen('Imagenes/pruebas/prueba8.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))

print("\n9_Esta imagen es un 3")
predictions = procesar_imagen('Imagenes/pruebas/prueba9.jpg', classifier)
print(predictions)
print("Los elementos que hay son:" + clasificar(predictions))