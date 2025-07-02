import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
img = cv2.imread("nasa.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if img is None:
    raise ValueError("Error: No se pudo cargar la imagen.")

# Obtener dimensiones de la imagen
height, width = img.shape

# Tamaño del bloque
block_size = 8

# Crear una copia de la imagen en RGB para dibujar líneas de la cuadrícula en color
img_grid = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Dibujar líneas verticales y horizontales para dividir en bloques de 8x8
for i in range(0, height, block_size):
    cv2.line(img_grid, (0, i), (width, i), (0, 255, 0), 1)  # Líneas horizontales

for j in range(0, width, block_size):
    cv2.line(img_grid, (j, 0), (j, height), (0, 255, 0), 1)  # Líneas verticales

# Mostrar la imagen con la cuadrícula
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para Matplotlib
plt.axis("off")
plt.show()

