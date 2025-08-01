import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
import os

# === FUNCIONES AUXILIARES ===
def calcular_mse_psnr_ssim(original, reconstruida):
    mse_val = mse(original, reconstruida)
    psnr_val = psnr(original, reconstruida, data_range=255)
    ssim_val = ssim(original, reconstruida, data_range=255)
    return mse_val, psnr_val, ssim_val

def calcular_histograma(imagen):
    hist, bins = np.histogram(imagen.flatten(), bins=256, range=[0, 256])
    return hist, bins[:-1]

# === DCT ===
def compress_dct(dct_matrix, percentage):
    height, width = dct_matrix.shape
    total_coeffs = height * width
    keep = int(total_coeffs * (percentage / 100))
    flat_indices = np.unravel_index(
        np.argsort(np.abs(dct_matrix).ravel())[::-1],
        dct_matrix.shape
    )
    mask = np.zeros_like(dct_matrix)
    mask[flat_indices[0][:keep], flat_indices[1][:keep]] = 1
    return dct_matrix * mask

# === DWT ===
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

# === CARGAR IMAGEN ===
imagen = cv2.imread("nasa.jpg", cv2.IMREAD_GRAYSCALE)
if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen 'nasa.jpg'")
imagen = cv2.resize(imagen, (512, 512))

# === DCT: Varios niveles de compresión ===
dct_full = cv2.dct(np.float32(imagen))
percentajes = [50, 25, 12.5, 6.25]
dct_imgs = []
mse_dct_vals = []
psnr_dct_vals = []
ssim_dct_vals = []

for p in percentajes:
    dct_compressed = compress_dct(dct_full, p)
    rec = np.uint8(np.clip(cv2.idct(dct_compressed), 0, 255))
    dct_imgs.append(rec)
    mse_val, psnr_val, ssim_val = calcular_mse_psnr_ssim(imagen, rec)
    mse_dct_vals.append(mse_val)
    psnr_dct_vals.append(psnr_val)
    ssim_dct_vals.append(ssim_val)

image_dct_50 = dct_imgs[0]
image_dct_12_5 = dct_imgs[2]
mse_dct_50 = mse_dct_vals[0]
psnr_dct_50 = psnr_dct_vals[0]
ssim_dct_50 = ssim_dct_vals[0]
mse_dct_12_5 = mse_dct_vals[2]
psnr_dct_12_5 = psnr_dct_vals[2]
ssim_dct_12_5 = ssim_dct_vals[2]

# === DWT ===
mse_dwt_vals = []
psnr_dwt_vals = []
ssim_dwt_vals = []
original_norm = imagen.astype(np.float32) / 255.0
rec_dwt_imgs = []

for nivel in range(1, 5):
    LL, coef = aplicar_dwt(imagen, nivel)
    rec_dwt = np.clip(reconstruir_idwt(LL, coef), 0, 255).astype(np.uint8)
    rec_dwt_imgs.append(rec_dwt)
    mse_val, psnr_val, ssim_val = calcular_mse_psnr_ssim(imagen, rec_dwt)
    mse_dwt_vals.append(mse_val)
    psnr_dwt_vals.append(psnr_val)
    ssim_dwt_vals.append(ssim_val)

rec_dwt_1 = rec_dwt_imgs[0]
rec_dwt_3 = rec_dwt_imgs[2]
mse_dwt_1 = mse_dwt_vals[0]
psnr_dwt_1 = psnr_dwt_vals[0]
ssim_dwt_1 = ssim_dwt_vals[0]
mse_dwt_3 = mse_dwt_vals[2]
psnr_dwt_3 = psnr_dwt_vals[2]
ssim_dwt_3 = ssim_dwt_vals[2]

# === PRIMERA HOJA ===
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 6))

for ax, img, title, mse_val, psnr_val, ssim_val in zip(
    axes1,
    [imagen, image_dct_50, image_dct_12_5],
    ["Original", "Compresión al 50% - DCT", "Compresión al 12.5% - DCT"],
    ["-", f"{mse_dct_50:.2f}", f"{mse_dct_12_5:.2f}"],
    ["-", f"{psnr_dct_50:.2f} dB", f"{psnr_dct_12_5:.2f} dB"],
    ["-", f"{ssim_dct_50:.4f}", f"{ssim_dct_12_5:.4f}"]):

    ax.imshow(img, cmap='gray')
    ax.set_title(f"{title}\nMSE: {mse_val} | PSNR: {psnr_val} | SSIM: {ssim_val}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# === SEGUNDA HOJA ===
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

for ax, img, title in zip(axes2, [imagen, image_dct_50, image_dct_12_5], ["Original", "Compresión al 50% - DCT", "Compresión al 12.5% - DCT"]):
    hist, bins = calcular_histograma(img)
    ax.bar(bins, hist, width=1, color='black')
    ax.set_title(f"Histograma - {title}")
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# === TERCERA HOJA ===
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 6))

for ax, img, title, mse_val, psnr_val, ssim_val in zip(
    axes3,
    [imagen, rec_dwt_1, rec_dwt_3],
    ["Original", "Compresión al 50% (j=1) - DWT de Haar", "Compresión al 12.5% (j=3) - DWT de Haar"],
    ["-", f"{mse_dwt_1:.2f}", f"{mse_dwt_3:.2f}"],
    ["-", f"{psnr_dwt_1:.2f} dB", f"{psnr_dwt_3:.2f} dB"],
    ["-", f"{ssim_dwt_1:.4f}", f"{ssim_dwt_3:.4f}"]):

    ax.imshow(img, cmap='gray')
    ax.set_title(f"{title}\nMSE: {mse_val} | PSNR: {psnr_val} | SSIM: {ssim_val}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# === CUARTA HOJA ===
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

for ax, img, title in zip(axes4, [imagen, rec_dwt_1, rec_dwt_3], ["Original", "Compresión al 50% (j=1) - DWT de Haar", "Compresión al 12.5% (j=3) - DWT de Haar"]):
    hist, bins = calcular_histograma(img)
    ax.bar(bins, hist, width=1, color='black')
    ax.set_title(f"Histograma - {title}")
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# === QUINTA HOJA ===
def annotate(ax, x, y, fmt="{:.2f}"):
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

fig5, axs5 = plt.subplots(1, 3, figsize=(18, 5))
labels = ["50%", "25%", "12.5%", "6.25%"]
x = [50, 25, 12.5, 6.25]

# MSE
axs5[0].plot(x, mse_dct_vals, marker='o', label='DCT + Cuant.', color='blue')
axs5[0].plot(x, mse_dwt_vals, marker='s', label='DWT Haar', color='red')
axs5[0].set_title('Compresi\u00f3n vs MSE')
axs5[0].set_xlabel('Nivel de compresi\u00f3n (%)')
axs5[0].set_ylabel('MSE')
axs5[0].invert_xaxis()
annotate(axs5[0], x, mse_dct_vals)
annotate(axs5[0], x, mse_dwt_vals)
axs5[0].legend()

# PSNR
axs5[1].plot(x, psnr_dct_vals, marker='o', label='DCT + Cuant.', color='blue')
axs5[1].plot(x, psnr_dwt_vals, marker='s', label='DWT Haar', color='red')
axs5[1].set_title('Compresi\u00f3n vs PSNR (dB)')
axs5[1].set_xlabel('Nivel de compresi\u00f3n (%)')
axs5[1].set_ylabel('PSNR (dB)')
axs5[1].invert_xaxis()
annotate(axs5[1], x, psnr_dct_vals)
annotate(axs5[1], x, psnr_dwt_vals)
axs5[1].legend()

# SSIM
axs5[2].plot(x, ssim_dct_vals, marker='o', label='DCT + Cuant.', color='blue')
axs5[2].plot(x, ssim_dwt_vals, marker='s', label='DWT Haar', color='red')
axs5[2].set_title('Compresi\u00f3n vs SSIM')
axs5[2].set_xlabel('Nivel de compresi\u00f3n (%)')
axs5[2].set_ylabel('SSIM')
axs5[2].invert_xaxis()
annotate(axs5[2], x, ssim_dct_vals, fmt="{:.4f}")
annotate(axs5[2], x, ssim_dwt_vals, fmt="{:.4f}")
axs5[2].legend()

plt.tight_layout()
plt.show()
