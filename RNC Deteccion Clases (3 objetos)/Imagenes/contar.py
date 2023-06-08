import os

path = 'Imagenes/test/3'

#path = 'Imagenes/train/3/'  # reemplaza 'ruta/a/carpeta' por la ruta de la carpeta que quieres contar

num_archivos = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

print(f"La carpeta '{path}' contiene {num_archivos} archivos.")
