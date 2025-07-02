import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Función para calcular el error cuadrático medio (MSE)
def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

# Función para calcular PSNR
def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

# Compresión por DCT reteniendo un porcentaje de coeficientes más significativos
def compress_dct(dct_matrix, percentage):
    height, width = dct_matrix.shape
    total_coeffs = height * width
    keep = int(total_coeffs * (percentage / 100))

    # Ordenar por magnitud absoluta (mayores primero)
    flat_indices = np.unravel_index(
        np.argsort(np.abs(dct_matrix).ravel())[::-1],
        dct_matrix.shape
    )
    
    mask = np.zeros_like(dct_matrix)
    mask[flat_indices[0][:keep], flat_indices[1][:keep]] = 1
    return dct_matrix * mask

# Procesamiento principal
def process_image(image_path="nasa.jpg"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica el archivo.")
    
    # Redimensionar para uniformidad
    image_resized = cv2.resize(image, (1024, 1024))

    # DCT de la imagen
    dct_full = cv2.dct(np.float32(image_resized))

    # Compresión conservando distintos porcentajes
    dct_50 = compress_dct(dct_full, 50)
    dct_12_5 = compress_dct(dct_full, 12.5)

    # Reconstrucción por IDCT
    image_50 = np.uint8(np.clip(cv2.idct(dct_50), 0, 255))
    image_12_5 = np.uint8(np.clip(cv2.idct(dct_12_5), 0, 255))

    # Cálculo de métricas
    metrics = [
        ("Original", image_resized, "-", "-", "-"),
        ("DCT 50%", image_50,
         f"{mse(image_resized, image_50):.2f}",
         f"{psnr(image_resized, image_50):.2f} dB",
         f"{ssim(image_resized, image_50, data_range=255):.4f}"),
        ("DCT 12.5%", image_12_5,
         f"{mse(image_resized, image_12_5):.2f}",
         f"{psnr(image_resized, image_12_5):.2f} dB",
         f"{ssim(image_resized, image_12_5, data_range=255):.4f}")
    ]

    # Visualización
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, (label, img, mse_val, psnr_val, ssim_val) in enumerate(metrics):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"{label}\nMSE: {mse_val} | PSNR: {psnr_val} | SSIM: {ssim_val}")

        axes[i, 1].hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.8)
        axes[i, 1].set_xlim([0, 256])
        axes[i, 1].set_title(f"Histograma - {label}")
        axes[i, 1].set_xlabel("Intensidad")
        axes[i, 1].set_ylabel("Frecuencia")

    fig.suptitle("Compresión de Imagen usando DCT y Evaluación de Calidad", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Ejecutar
process_image("nasa.jpg")
