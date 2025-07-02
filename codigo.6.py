import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('nasa.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar la DWT de nivel 1 (Wavelet 'haar')
LL, (LH, HL, HH) = pywt.dwt2(imagen, 'haar')

# Mostrar solo las cuatro subbandas en formato 2x2
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Subbanda LL (Baja Frecuencia, Aproximación) - Arriba Izquierda
axes[0, 0].imshow(LL, cmap='gray')
axes[0, 0].set_title('LL1 (Aproximación)')
axes[0, 0].axis('off')

# Subbanda HL (Detalles Verticales) - Arriba Derecha
axes[0, 1].imshow(HL, cmap='gray')
axes[0, 1].set_title('HL1 (Detalles Verticales)')
axes[0, 1].axis('off')

# Subbanda LH (Detalles Horizontales) - Abajo Izquierda
axes[1, 0].imshow(LH, cmap='gray')
axes[1, 0].set_title('LH1 (Detalles Horizontales)')
axes[1, 0].axis('off')

# Subbanda HH (Detalles Diagonales) - Abajo Derecha
axes[1, 1].imshow(HH, cmap='gray')
axes[1, 1].set_title('HH1 (Detalles Diagonales)')
axes[1, 1].axis('off')

# Mostrar todas las subbandas en 2x2 sin la imagen original
plt.show()

# Guardar las imágenes de las subbandas en formato JPEG2000 (.jp2)
cv2.imwrite('LL_n1.jp2', LL, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('LH_n1.jp2', LH, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('HL_n1.jp2', HL, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('HH_n1.jp2', HH, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

print("Subbandas guardadas como nasa_LL.jp2, nasa_LH.jp2, nasa_HL.jp2, nasa_HH.jp2")
