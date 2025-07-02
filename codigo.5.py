import numpy as np
from scipy.fftpack import dct

# === 1. Matriz original ===
original_matrix = np.array([
    [29, -5, -3, -12, 34, 8, -30, -59],
    [-7, -18, -22, 14, 51, 50, -5, -47],
    [-28, -21, -13, -12, 25, 50, 25, -36],
    [-35, -36, -22, -31, -16, -6, -1, -32],
    [-39, -62, -49, -1, -27, -57, -57, -22],
    [-9, -50, -53, -11, -35, -77, -68, -11],
    [47, -2, -41, -45, -38, -48, -34, -5],
    [78, 30, -39, -35, -20, 0, -12, 1]
], dtype=float)

# === 2. Función para DCT 2D ===
def dct2(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

# === 3. Matriz de cuantización estándar JPEG (50%) ===
Q50 = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

# === 4. Matriz de cuantización al 12.5% (de tu imagen) ===
Q125 = np.array([
    [64, 44, 40, 64, 96, 160, 204, 244],
    [48, 48, 56, 76, 104, 232, 240, 220],
    [56, 52, 64, 96, 160, 228, 276, 224],
    [56, 68, 88, 116, 204, 348, 320, 248],
    [72, 88, 148, 224, 272, 436, 412, 308],
    [96, 140, 220, 256, 324, 416, 452, 368],
    [196, 256, 312, 348, 412, 484, 480, 404],
    [288, 368, 380, 392, 448, 400, 412, 396]
])

# === 5. Aplicar DCT ===
dct_matrix = dct2(original_matrix)

# === 6. Cuantización ===
quant_50 = np.round(dct_matrix / Q50).astype(int)
quant_125 = np.round(dct_matrix / Q125).astype(int)

# === 7. Mostrar matrices ===
np.set_printoptions(linewidth=160)

# ANSI bold formatting
bold = "\033[1m"
reset = "\033[0m"

print(f"{bold} Matriz DCT (redondeada):{reset}\n", np.round(dct_matrix).astype(int))
print(f"\n{bold} DCT Cuantizada con Q50 (50%):{reset}\n", quant_50)
print(f"\n{bold} DCT Cuantizada con Q12.5 (12.5%):{reset}\n", quant_125)
