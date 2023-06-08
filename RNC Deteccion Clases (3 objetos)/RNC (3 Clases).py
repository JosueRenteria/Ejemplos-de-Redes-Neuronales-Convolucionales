# Libreria para inicializar la Red Neuronal y los pesos aleatorios.
from keras.models import Sequential
# Libreria para crear una capa de Convolucion.
from keras.layers import Conv2D
# Libreria del Max Pooling.
from keras.layers import MaxPooling2D
# Libreria para el aplanado.
from keras.layers import Flatten
# Libreria para crear la sinapsis.
from keras.layers import Dense
# Libreria para la capa de sobreajuste.
from keras.layers import Dropout
# librerias para el pre-procesamiento.
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.preprocessing.image import ImageDataGenerator
# Libreria para graficar.
import matplotlib.pyplot as plt
# Libreria de Obtimizador.
from keras.optimizers import Adam

# Inicializacion del Modelo.
classifier = Sequential()

# ------------Paso 1 - Convolución------------
# Creamos la capa de la operacion de Convolucion para hacer los mapas de Caracteristicas..

# Recordemos que en la primera capa hay que indicar el tamaño de la entrada.
classifier.add(Conv2D(32, (4, 4), activation='relu', input_shape=(64, 64, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Capa 2 Para el Mapa de Caracteristicas.
classifier.add(Conv2D(64, (4, 4), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Antes de poder pasar de la capa convolucional a la densa, es necesario "aplanar" la salida, por eso se usa Flatten
classifier.add(Flatten())

# ------------Paso 2 - Red Neuronal------------
# Primera Capa.
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.3))

# Capa de Salida.
classifier.add(Dense(3, activation='softmax'))

# Compilador con el optimizador,
adam = Adam(lr=0.001)
classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# ------------Paso 3 - Ajustar la CNN a las imágenes para entrenar------------
# Limpiar la imagen y no tener sobre ajuste (transformaciones de las imagenes).
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=20,
        zoom_range=0.2,
        #preprocessing_function=segment_image,  # Aplicar la segmentación a las imágenes
        horizontal_flip=True)

# Limpiar la imagen y no tener sobre ajuste (transformaciones de las imagenes).
test_datagen = ImageDataGenerator(rescale=1./255)

# Para la parte de los datos de training.
training_dataset = train_datagen.flow_from_directory('imagenes/train/',
                                                target_size=(64, 64),
                                                batch_size=2,
                                                class_mode='categorical')
# Para los datos de testing.
testing_dataset = test_datagen.flow_from_directory('imagenes/test/',
                                                target_size=(64, 64),
                                                batch_size=2,
                                                class_mode='categorical')

# Mostramos los Datsos de training y teasting.
print(len(training_dataset))
print(len(testing_dataset))

# ------------Paso 4 - Compilador y definicion de las Epocas y imagenes------------

# Paso 4 - Compilador y definición de las épocas e imágenes
history = classifier.fit(
    training_dataset, # Conjunto de entrenamiento
    steps_per_epoch=len(training_dataset), # Muestras que toma en cada ciclo de entrenamiento, pasaremos todas las imágenes
    epochs=15, # Cuántas épocas usaremos para entrenar
    validation_data=testing_dataset, # Conjunto de validación
    validation_steps=len(testing_dataset) # Cada cuántos pasos validaremos nuestro resultado en este caso 2 cada 8 épocas
)

# Mostramos la evolución del loss (pérdida) a través de los epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'])
plt.title('Cross Entropy')
plt.show()

# Mostramos la evolución del accuracy (precisión) a través de los epochs
plt.plot(history.history['accuracy'])  # 'acc' fue cambiado a 'accuracy' en Keras 2.x
plt.plot(history.history['val_accuracy'])  # 'val_acc' fue cambiado a 'val_accuracy' en Keras 2.x
plt.legend(['train', 'valid'])
plt.title('Accuracy')
plt.show()
