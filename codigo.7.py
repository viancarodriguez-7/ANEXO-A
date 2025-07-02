import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('nasa.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar la DWT de nivel 1
LL1, (LH1, HL1, HH1) = pywt.dwt2(imagen, 'haar')

# Aplicar la DWT de nivel 2 sobre LL1
LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')

# Aplicar la DWT de nivel 3 sobre LL2 (subbandas finales)
LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

# Mostrar solo las cuatro subbandas del tercer nivel en formato 2x2
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Subbanda LL3 (Aproximación - Baja Frecuencia) - Arriba Izquierda
axes[0, 0].imshow(LL3, cmap='gray')
axes[0, 0].set_title('LL3 (Aproximación)')
axes[0, 0].axis('off')

# Subbanda HL3 (Detalles Verticales) - Arriba Derecha
axes[0, 1].imshow(HL3, cmap='gray')
axes[0, 1].set_title('HL3 (Detalles Verticales)')
axes[0, 1].axis('off')

# Subbanda LH3 (Detalles Horizontales) - Abajo Izquierda
axes[1, 0].imshow(LH3, cmap='gray')
axes[1, 0].set_title('LH3 (Detalles Horizontales)')
axes[1, 0].axis('off')

# Subbanda HH3 (Detalles Diagonales) - Abajo Derecha
axes[1, 1].imshow(HH3, cmap='gray')
axes[1, 1].set_title('HH3 (Detalles Diagonales)')
axes[1, 1].axis('off')

# Mostrar todas las subbandas en 2x2 sin la imagen original
plt.show()

# Guardar las imágenes de las subbandas en formato JPEG2000 (.jp2)
cv2.imwrite('LL3.jp2', LL3, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('LH3.jp2', LH3, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('HL3.jp2', HL3, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
cv2.imwrite('HH3.jp2', HH3, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

print("Subbandas del tercer nivel guardadas como nasa_LL3.jp2, nasa_LH3.jp2, nasa_HL3.jp2, nasa_HH3.jp2")
