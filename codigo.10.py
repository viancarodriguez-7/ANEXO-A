import numpy as np

# Función inversa de Haar 2D
def inverse_haar2d(LL, HL, LH, HH):
    h, w = LL.shape
    result = np.zeros((h * 2, w * 2))
    for i in range(h):
        for j in range(w):
            a = LL[i, j]
            b = HL[i, j]
            c = LH[i, j]
            d = HH[i, j]
            result[2*i, 2*j]     = a + b + c + d
            result[2*i, 2*j+1]   = a - b + c - d
            result[2*i+1, 2*j]   = a + b - c - d
            result[2*i+1, 2*j+1] = a - b - c + d
    return result

# === Parte 1: Reconstrucción desde nivel 3 ===

# Nivel 3
LL3 = np.array([[112]])
HL3 = np.array([[0]])
LH3 = np.array([[9]])
HH3 = np.array([[-8]])

LL_sub2 = inverse_haar2d(LL3, HL3, LH3, HH3).round().astype(int)

# Nivel 2
HL2 = np.array([[0, 24], [17, -6]])
LH2 = np.array([[10, 0], [-17, -12]])
HH2 = np.array([[5, 12], [-22, 1]])

LL_sub1 = inverse_haar2d(LL_sub2, HL2, LH2, HH2).round().astype(int)

# Nivel 1
HL1 = np.array([[11, -12, 7, 18],
                [-2, 2, -9, 23],
                [16, -22, 18, -23],
                [24, 0, 2, -10]])
LH1 = np.array([[12, -7, -15, -9],
                [6, 7, 24, 6],
                [-10, 4, 7, 0],
                [-16, -3, -16, -7]])
HH1 = np.array([[6, 6, 6, -3],
                [-2, -2, -4, 8],
                [-4, -2, -3, 6],
                [0, 2, 8, -4]])

reconstructed_12_5 = inverse_haar2d(LL_sub1, HL1, LH1, HH1).round().astype(int)

# === Parte 2: Reconstrucción desde una matriz LL1 dada directamente ===

LL1_new = np.array([
    [128, 117, 164, 93],
    [98, 109, 141, 117],
    [88, 100, 166, 88],
    [79, 88, 102, 116]
])
HL1_new = np.array([
    [11, -12, 7, 18],
    [-2, 2, -9, 23],
    [16, -22, 18, -23],
    [24, 0, -2, -10]
])
LH1_new = np.array([
    [12, -7, -15, -9],
    [6, 7, 24, 6],
    [-10, 4, 7, 0],
    [-16, -3, -16, -7]
])
HH1_new = np.array([
    [6, 6, 6, -3],
    [-2, -2, -4, 8],
    [-4, -2, -3, 6],
    [0, 2, 8, -4]
])

reconstructed_LL1_only = inverse_haar2d(LL1_new, HL1_new, LH1_new, HH1_new).round().astype(int)

# === Resultados ===

bold = lambda s: f"\033[1m{s}\033[0m"

print(bold("\n\t\tReconstrucción al 12.5% de compresión\n"))

print(bold(" LL_2 (recontrucción al nivel 2):"))
print(LL_sub2)

print(bold("\n LL_1 (recontrucción al nivel 1):"))
print(LL_sub1)

print(bold("\n Matriz final reconstruida:"))
print(reconstructed_12_5)


print(bold("\n\n\t\tReconstrucción al 50% de compresión"))
print(bold("\n Matriz final reconstruida:"))
print(reconstructed_LL1_only)
