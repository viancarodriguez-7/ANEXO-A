import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz original (sin compresión)
original_matrix = np.array([
    [29, -5, -3, -12, 34, 8, -30, -59],
    [-7, -18, -22, 14, 51, 50, -5, -47],
    [-28, -21, -13, -12, 25, 50, 25, -36],
    [-35, -36, -22, -31, -16, -6, -1, -32],
    [-39, -62, -49, -1, -27, -57, -57, -22],
    [-9, -50, -53, -11, -35, -77, -68, -11],
    [47, -2, -41, -45, -38, -48, -34, -5],
    [78, 30, -39, -35, -20, 0, -12, 1]
])

# Matriz de compresión al 50%
quantization_50 = np.array([
    [-8,  2,  0,  5,  2,  0,  0,  0],
    [ 4, -3, -9,  2, -2,  0,  0,  0],
    [ 6,  5,  2,  2,  0,  0,  0,  0],
    [-6,  2,  0, -2,  1,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
])

# Matriz de compresión al 12.5%
quantization_12_5 = np.array([
    [-2,  1,  0,  1,  0,  0,  0,  0],
    [ 1, -1, -2,  1,  0,  0,  0,  0],
    [ 2,  1,  1,  1,  0,  0,  0,  0],
    [-1,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
])

# Visualización de las matrices con mapas de calor
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

vmin, vmax = -128, 127  # Rango fijo para los colores

sns.heatmap(original_matrix, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[0], vmin=vmin, vmax=vmax)
axes[0].set_title("Matriz Original (sin compresión)")

sns.heatmap(quantization_50, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[1], vmin=vmin, vmax=vmax)
axes[1].set_title("Matriz de compresión al 50%")

sns.heatmap(quantization_12_5, annot=True, cmap="gray", fmt="d", linewidths=0.5, ax=axes[2], vmin=vmin, vmax=vmax)
axes[2].set_title("Matriz de compresión al 12.5%")

plt.show()
