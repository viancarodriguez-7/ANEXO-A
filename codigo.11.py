import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz original
original_matrix = np.array([
    [157, 123, 104, 116, 162, 136, 99, 69],
    [121, 111, 106, 142, 180, 178, 123, 81],
    [100, 108, 123, 109, 163, 167, 136, 110],
    [92,  92, 113, 91,  123, 111, 108, 114],
    [67,  87, 101, 107, 83,  89,  91, 87],
    [97, 101, 97,  95,  75,  69,  79, 99],
    [163, 137, 82,  88,  104, 66,  97, 121],
    [195, 169, 84,  98,  122, 116, 119, 127]
])

# Matriz reconstruida con 25% de compresión (DEBÍA LLAMARSE reconstructed_50)
reconstructed_25 = np.array([
    [151, 124, 113, 132, 149, 134, 98, 72],
    [112, 109, 115, 141, 175, 176, 125, 69],
    [106, 105, 101, 114, 154, 180, 149, 96],
    [103, 94,  89,  99,  119, 130, 122, 108],
    [83,  71,  87,  117, 109, 74,  70, 96],
    [117, 79,  76,  103, 89,  49,  64, 119],
    [184, 120, 77,  82,  85,  75,  97, 141],
    [213, 149, 97,  98,  120, 124, 119, 118]
])

# Matriz reconstruida con 12.5% de compresión
reconstructed_12_5 = np.array([
    [184, 134,  96, 117, 167, 174, 117,  54],
    [142, 114,  99, 124, 165, 171, 130,  83],
    [89,  91,  103, 128, 150, 151, 130, 108],
    [61,  81,  107, 120, 119, 110, 105, 104],
    [73,  92,  109, 105,  86,  72,  74,  83],
    [117, 117, 109,  91,  72,  64,  70,  80],
    [167, 143, 109,  86,  81,  89, 101, 108],
    [199, 158, 109,  86,  96, 119, 135, 140]
])

# Definir el mínimo y máximo de la escala de color
vmin = 0
vmax = 255

# Crear la figura y subgráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Graficar las matrices
sns.heatmap(original_matrix, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[0], vmin=vmin, vmax=vmax)
axes[0].set_title("Matriz Original (sin compresión)")
axes[0].set_xlabel("Columnas")
axes[0].set_ylabel("Filas")

sns.heatmap(reconstructed_25, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[1], vmin=vmin, vmax=vmax)
axes[1].set_title("DCT reconstruida (50% de compresión)")
axes[1].set_xlabel("Columnas")
axes[1].set_ylabel("Filas")

sns.heatmap(reconstructed_12_5, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[2], vmin=vmin, vmax=vmax)
axes[2].set_title("DCT reconstruida (12.5% de compresión)")
axes[2].set_xlabel("Columnas")
axes[2].set_ylabel("Filas")

# Ajustar la distribución y mostrar
plt.tight_layout()
plt.show()
