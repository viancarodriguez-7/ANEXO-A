import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definir las matrices
original = np.array([
    [157, 123, 105, 116, 162, 136, 98, 69],
    [121, 110, 106, 142, 179, 178, 123, 81],
    [100, 107, 115, 116, 153, 178, 153, 92],
    [93, 92, 106, 97, 112, 122, 127, 96],
    [89, 66, 79, 127, 101, 71, 71, 106],
    [119, 78, 75, 117, 93, 51, 60, 117],
    [175, 126, 87, 83, 90, 80, 94, 123],
    [206, 158, 89, 93, 108, 128, 116, 129]
])

compression_50 = np.array([
    [157, 123, 104, 116, 162, 136, 99, 69],
    [121, 111, 106, 142, 180, 178, 123, 81],
    [100, 108, 116, 116, 152, 178, 154, 92],
    [92, 92, 106, 98, 112, 122, 126, 96],
    [90, 66, 80, 128, 188, 158, 71, 105],
    [118, 78, 76, 116, 180, 138, 59, 117],
    [87, 39, 87, 83, 92, 80, 95, 123],
    [119, 71, 89, 93, 108, 128, 117, 129]
])

compression_12_5 = np.array([
    [157, 123, 105, 117, 163, 137, 99, 69],
    [121, 111, 107, 143, 181, 179, 123, 81],
    [100, 108, 115, 115, 152, 178, 154, 92],
    [92, 92, 105, 97, 112, 122, 126, 96],
    [91, 67, 79, 127, 100, 70, 71, 105],
    [119, 79, 75, 115, 92, 50, 59, 117],
    [175, 127, 88, 84, 94, 74, 93, 121],
    [207, 159, 90, 94, 110, 122, 115, 127]
])

# Función para graficar heatmaps con escala entre 0 y 255
def plot_heatmaps(matrices, titles):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, matrix, title in zip(axes, matrices, titles):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="gray", linewidths=0.5, ax=ax, vmin=0, vmax=255)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

# Graficar los heatmaps
plot_heatmaps(
    [original, compression_50, compression_12_5], 
    ["Matriz Original (sin compresión)", "DWT reconstruida (50% de compresión)", "DWT reconstruida (12.5% de compresión)"]
)
