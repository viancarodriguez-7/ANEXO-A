import numpy as np

def haar_2d_transform(matrix):
    """Aplica una DWT 2D de Haar al nivel 1 manualmente."""
    h, w = matrix.shape
    assert h % 2 == 0 and w % 2 == 0, "La matriz debe tener dimensiones pares"
    
    LL = np.zeros((h//2, w//2))
    LH = np.zeros((h//2, w//2))
    HL = np.zeros((h//2, w//2))
    HH = np.zeros((h//2, w//2))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            a = matrix[i, j]
            b = matrix[i, j+1]
            c = matrix[i+1, j]
            d = matrix[i+1, j+1]
            
            LL[i//2, j//2] = (a + b + c + d) / 4
            LH[i//2, j//2] = (a + b - c - d) / 4
            HL[i//2, j//2] = (a - b + c - d) / 4
            HH[i//2, j//2] = (a - b - c + d) / 4

    return LL, LH, HL, HH

def formatear_matriz(matriz: np.ndarray, ancho: int = 6) -> list[list[str]]:
    """Formatea con espacio uniforme y alineación derecha."""
    return [[f"{int(round(e)):>{ancho}}" for e in fila] for fila in matriz]

def imprimir_subbandas(LL, LH, HL, HH, nivel):
    """Imprime las subbandas con alineación perfecta en tabla rectangular."""
    ancho = 6
    LL_fmt = formatear_matriz(LL, ancho)
    HL_fmt = formatear_matriz(HL, ancho)
    LH_fmt = formatear_matriz(LH, ancho)
    HH_fmt = formatear_matriz(HH, ancho)

    filas = len(LL_fmt)
    columnas = len(LL_fmt[0])

    # Negrita ANSI
    negrita = lambda txt: f"\033[1m{txt}\033[0m"

    print(f"\n{negrita(f' Nivel {nivel}')}\n")

    print(f"{'[LL_' + str(nivel) + ']':<{(ancho + 1) * columnas + 2}}  {'[HL_' + str(nivel) + ']'}")
    for i in range(filas):
        fila_LL = ' '.join(LL_fmt[i])
        fila_HL = ' '.join(HL_fmt[i])
        print(f"[ {fila_LL} ]   [ {fila_HL} ]")

    print()
    print(f"{'[LH_' + str(nivel) + ']':<{(ancho + 1) * columnas + 2}}  {'[HH_' + str(nivel) + ']'}")
    for i in range(filas):
        fila_LH = ' '.join(LH_fmt[i])
        fila_HH = ' '.join(HH_fmt[i])
        print(f"[ {fila_LH} ]   [ {fila_HH} ]")

def mostrar_dwt_manual_en_niveles(matriz: np.ndarray, niveles: int = 3):
    """Aplica la DWT Haar manual y muestra los resultados alineados por nivel."""
    actual = matriz.copy()
    for nivel in range(1, niveles + 1):
        LL, LH, HL, HH = haar_2d_transform(actual)
        imprimir_subbandas(LL, LH, HL, HH, nivel)
        actual = LL

# Matriz original
original_matrix = np.array([
    [157, 123, 105, 116, 162, 136, 98, 69],
    [121, 110, 106, 142, 179, 178, 123, 81],
    [100, 107, 115, 116, 153, 178, 153, 92],
    [93,  92, 106,  97, 112, 122, 127, 96],
    [89,  66,  79, 127, 101, 71,  71, 106],
    [119, 78,  75, 117, 93,  51,  60, 117],
    [175, 126, 87,  83,  90, 80,  94, 123],
    [206, 158, 89,  93, 108, 128, 116, 129]
], dtype=float)

# Ejecutar transformada manual alineada
mostrar_dwt_manual_en_niveles(original_matrix, niveles=3)
