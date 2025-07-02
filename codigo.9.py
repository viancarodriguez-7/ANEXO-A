import numpy as np
from scipy.fftpack import idct

# Función IDCT 2D
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ANSI para negrita en consola
bold = "\033[1m"
reset = "\033[0m"

# === 1. Matriz de cuantización JPEG estándar al 50% ===
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
])

# === 2. Matriz de cuantización al 12.5% ===
Q125 = np.array([
    [64, 44, 40, 64, 96, 160, 204, 244],
    [48, 48, 56, 76, 104, 232, 240, 220],
    [56, 52, 64, 96, 160, 228, 276, 224],
    [56, 68, 88,116, 204, 348, 320, 248],
    [72, 88,148,224, 272, 436, 412, 308],
    [96,140,220,256, 324, 416, 452, 368],
    [196,256,312,348, 412, 484, 480, 404],
    [288,368,380,392, 448, 400, 412, 396]
])

# === 3. Coeficientes cuantizados ===
quantization_50 = np.array([
    [-8,  2,  0,  5,  2,  0,  0,  0],
    [ 4, -3, -9,  2, -2,  0,  0,  0],
    [ 6,  5,  2,  2,  0,  0,  0,  0],
    [-6,  2,  0, -2,  1,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
], dtype=float)

quantization_12_5 = np.array([
    [-2,  1,  0,  1,  0,  0,  0,  0],
    [ 1, -1, -2,  1,  0,  0,  0,  0],
    [ 2,  1,  1,  1,  0,  0,  0,  0],
    [-1,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
], dtype=float)

# === 4. Dequantización ===
dequant_50 = quantization_50 * Q50
dequant_125 = quantization_12_5 * Q125

# === 5. Aplicar IDCT y sumar 128 (nivel de gris medio) ===
reconstructed_50 = np.round(idct2(dequant_50)).astype(int) + 128
reconstructed_125 = np.round(idct2(dequant_125)).astype(int) + 128

# === 6. Mostrar resultados ===
print(f"{bold}Matriz reconstruida al 50%:{reset}\n")
print(reconstructed_50)
print(f"\n{bold}Matriz reconstruida al 12.5%:{reset}\n")
print(reconstructed_125)
