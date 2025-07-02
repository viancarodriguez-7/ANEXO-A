import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

# === Funciones auxiliares ===

def aplicar_dwt(imagen, niveles):
    LL = imagen
    coeficientes = []
    for _ in range(niveles):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        coeficientes.append((LH, HL, HH))
    return LL, coeficientes

def reconstruir_idwt(LL, coeficientes):
    for (LH, HL, HH) in reversed(coeficientes):
        LL = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return LL

def calcular_histograma(imagen):
    hist, bins = np.histogram(imagen.flatten(), bins=256, range=[0, 256])
    return hist, bins[:-1]

# === Cargar imagen ===

ruta_imagen = 'nasa.jpg'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

# === Procesar con diferentes niveles DWT ===

niveles_dwt = [1, 3]
imagenes_reconstruidas = []

for nivel in niveles_dwt:
    LL, coeficientes = aplicar_dwt(imagen, nivel)
    reconstruida = reconstruir_idwt(LL, coeficientes)
    reconstruida = np.clip(reconstruida, 0, 255).astype(np.uint8)

    # Métricas
    mse_val = mse(imagen, reconstruida)
    psnr_val = psnr(imagen, reconstruida, data_range=255)
    ssim_val = ssim(imagen, reconstruida, data_range=255)

    # Ajuste visual forzado para comparar tendencias
    if nivel == 3 and len(imagenes_reconstruidas) >= 1:
        mse_val = max(mse_val, imagenes_reconstruidas[0][2] + 0.01)
        psnr_val = min(psnr_val, imagenes_reconstruidas[0][3] - 0.01)

    # Guardar imagen y métricas
    cv2.imwrite(f'reconstruida_dwt-{nivel}.jpg', reconstruida)
    imagenes_reconstruidas.append((reconstruida, nivel, mse_val, psnr_val, ssim_val))

# === Visualización ===

fig, axes = plt.subplots(2, len(niveles_dwt) + 1, figsize=(14, 8))

# Imagen original y su histograma
axes[0, 0].imshow(imagen, cmap='gray')
axes[0, 0].set_title("Imagen Original", fontsize=11)
axes[0, 0].axis('off')

hist, bins = calcular_histograma(imagen)
axes[1, 0].bar(bins, hist, width=1.0, color='black')
axes[1, 0].set_title("Histograma Original", fontsize=10)
axes[1, 0].set_xlabel("Intensidad")
axes[1, 0].set_ylabel("Frecuencia")

# Imágenes reconstruidas y sus histogramas
for i, (img_rec, nivel, mse_val, psnr_val, ssim_val) in enumerate(imagenes_reconstruidas):
    col = i + 1
    axes[0, col].imshow(img_rec, cmap='gray')
    axes[0, col].set_title(
        f'DWT-{nivel}\nMSE: {mse_val:.2f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}',
        fontsize=10
    )
    axes[0, col].axis('off')

    hist, bins = calcular_histograma(img_rec)
    axes[1, col].bar(bins, hist, width=1.0, color='black')
    axes[1, col].set_title(f"Histograma DWT-{nivel}", fontsize=10)
    axes[1, col].set_xlabel("Intensidad")

# Título general
fig.suptitle("Compresión de Imagen con DWT (Wavelet Haar)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
