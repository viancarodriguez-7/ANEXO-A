import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim

# === FUNCIONES AUXILIARES ===
def calcular_mse_psnr_ssim(original, comprimida):
    mse_val = np.mean((original - comprimida) ** 2)
    if mse_val == 0:
        psnr_val = float("inf")
    else:
        psnr_val = 10 * np.log10(255 ** 2 / mse_val)
    ssim_val = ssim(original, comprimida, data_range=255)
    return mse_val, psnr_val, ssim_val

def calcular_histograma(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist, bins[:-1]

def compress_tdc(tdc, percent):
    (h, w) = tdc.shape
    keep_h = int(h * percent / 100)
    keep_w = int(w * percent / 100)
    mask = np.zeros_like(tdc)
    mask[:keep_h, :keep_w] = 1
    return tdc * mask

def aplicar_tdw(img, niveles=1):
    LL = np.float32(img)
    coeficientes = []
    for _ in range(niveles):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        coeficientes.append((LH, HL, HH))
    return LL, coeficientes

def reconstruir_itdw(LL, coeficientes):
    for (LH, HL, HH) in reversed(coeficientes):
        LL = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return LL

def guardar_histograma(img, titulo, filename_img, filename_excel, filename_csv, output_dir="."):
    hist, bins = calcular_histograma(img)

    # Crear tabla
    df = pd.DataFrame({"Intensidad": bins.astype(int), "Frecuencia": hist})

    # Guardar tabla
    df.to_excel(os.path.join(output_dir, filename_excel), index=False)
    df.to_csv(os.path.join(output_dir, filename_csv), index=False)

    # Graficar y guardar imagen PNG
    plt.figure(figsize=(10,5))
    plt.bar(df["Intensidad"], df["Frecuencia"], width=1, color="black")
    plt.title(f"Histograma - {titulo}")
    plt.xlabel("Intensidad (0-255)")
    plt.ylabel("Frecuencia de píxeles")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename_img))
    plt.close()

# === CARGAR IMAGEN ===
imagen = cv2.imread("nasa.jpg", cv2.IMREAD_GRAYSCALE)
if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen 'nasa.jpg'. Revisa la ruta.")
imagen = cv2.resize(imagen, (512, 512))

output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# === HISTOGRAMA ORIGINAL ===
guardar_histograma(imagen, 
                   "Imagen Original", 
                   "hist_original.png", 
                   "tabla_hist_original.xlsx", 
                   "tabla_hist_original.csv", 
                   output_dir)

# === TDC: Varios niveles de compresión ===
tdc_full = cv2.dct(np.float32(imagen))
percentajes = [50, 12.5]  # Solo los que necesitas
tdc_imgs = {}

for p in percentajes:
    tdc_compressed = compress_tdc(tdc_full, p)
    rec = np.uint8(np.clip(cv2.idct(tdc_compressed), 0, 255))
    tdc_imgs[p] = rec

# Guardar histogramas TDC
guardar_histograma(tdc_imgs[50], 
                   "Compresión al 50% - TDC", 
                   "hist_tdc_50.png", 
                   "tabla_hist_tdc_50.xlsx", 
                   "tabla_hist_tdc_50.csv", 
                   output_dir)

guardar_histograma(tdc_imgs[12.5], 
                   "Compresión al 12,5% - TDC", 
                   "hist_tdc_12,5.png", 
                   "tabla_hist_tdc_12,5.xlsx", 
                   "tabla_hist_tdc_12,5.csv", 
                   output_dir)

# === TDW ===
rec_tdw_imgs = {}
niveles = {1: 50, 3: 12.5}  # niveles con % de compresión aproximada

for nivel, porcentaje in niveles.items():
    LL, coef = aplicar_tdw(imagen, nivel)
    rec_tdw = np.clip(reconstruir_itdw(LL, coef), 0, 255).astype(np.uint8)
    rec_tdw_imgs[porcentaje] = rec_tdw

# Guardar histogramas TDW
guardar_histograma(rec_tdw_imgs[50], 
                   "Compresión al 50% - TDW", 
                   "hist_tdw_50.png", 
                   "tabla_hist_tdw_50.xlsx", 
                   "tabla_hist_tdw_50.csv", 
                   output_dir)

guardar_histograma(rec_tdw_imgs[12.5], 
                   "Compresión al 12,5% - TDW", 
                   "hist_tdw_12,5.png", 
                   "tabla_hist_tdw_12,5.xlsx", 
                   "tabla_hist_tdw_12,5.csv", 
                   output_dir)

print("Histogramas y tablas (Excel + CSV) generados para: Original, TDC (50%, 12,5%) y TDW (50%, 12,5%).")
