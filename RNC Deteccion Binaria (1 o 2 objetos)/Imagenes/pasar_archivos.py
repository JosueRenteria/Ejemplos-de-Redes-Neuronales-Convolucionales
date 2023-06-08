import os
import random
import shutil

# Definir la ruta de origen y destino de las carpetas
ruta_origen = 'Imagenes/train/3/'
ruta_destino = 'Imagenes/test/3/'

# Obtener una lista de los nombres de archivo en la carpeta de origen
archivos = os.listdir(ruta_origen)

# Mezclar aleatoriamente los nombres de archivo
random.shuffle(archivos)

# Seleccionar los primeros 20 nombres de archivo de la lista mezclada
archivos_a_mover = archivos[:10]

# Mover los archivos seleccionados a la carpeta de destino
for archivo in archivos_a_mover:
    origen = os.path.join(ruta_origen, archivo)
    destino = os.path.join(ruta_destino, archivo)
    shutil.move(origen, destino)

print(f"Se han movido aleatoriamente {len(archivos_a_mover)} archivos de la carpeta de origen a la carpeta de destino.")
