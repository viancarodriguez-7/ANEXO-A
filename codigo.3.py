import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

# 1. Matriz original 8x8
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

# 2. Transformada DCT 2D
def dct2(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

dct_matrix = dct2(original_matrix)

# 3. Matriz redondeada para mapa de calor
dct_matrix_rounded = np.round(dct_matrix).astype(int)

# 4. Crear el mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(dct_matrix_rounded, annot=True, cmap="gray", fmt="d", linewidths=0.5, cbar=True, vmin=-128, vmax=127)
plt.title("Mapa de Calor de la Matriz DCT")
plt.xlabel("Coeficientes de frecuencia (Horizontal)")
plt.ylabel("Coeficientes de frecuencia (Vertical)")
plt.show()

# 5. Interpolación para superficie 3D
x = np.arange(8)
y = np.arange(8)
x_fine = np.linspace(0, 7, 100)
y_fine = np.linspace(0, 7, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

interp_dct = RectBivariateSpline(y, x, dct_matrix, kx=3, ky=3)
Z_dct_smooth = interp_dct(y_fine, x_fine)

# 6. Graficar superficies 3D con dos colormaps
fig = plt.figure(figsize=(14, 6))

# Superficie 1 - coolwarm
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_fine, Y_fine, Z_dct_smooth, cmap='coolwarm', edgecolor='none')
ax1.set_title('Matriz DCT - Colormap: coolwarm')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Intensidad')

# Superficie 2 - gray
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_fine, Y_fine, Z_dct_smooth, cmap='gray', edgecolor='none')
ax2.set_title('Matriz DCT - Colormap: gray')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Intensidad')

plt.tight_layout()
plt.show()
