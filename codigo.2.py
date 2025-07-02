import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

# 1. Matriz original 8x8
original_matrix = np.array([
    [157, 123, 105, 116, 162, 136, 98, 69],
    [121, 110, 106, 142, 179, 178, 123, 81],
    [100, 107, 115, 116, 153, 178, 153, 92],
    [93, 92, 106, 97, 112, 122, 127, 96],
    [89, 66, 79, 127, 101, 71, 71, 106],
    [119, 78, 75, 117, 93, 51, 60, 117],
    [175, 126, 87, 83, 90, 80, 94, 123],
    [206, 158, 89, 93, 108, 128, 116, 129]
], dtype=float)

# 4. Mallas originales y refinadas
x = np.arange(8)
y = np.arange(8)
x_fine = np.linspace(0, 7, 100)
y_fine = np.linspace(0, 7, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# 5. Interpolación suave con RectBivariateSpline
interp_orig = RectBivariateSpline(y, x, original_matrix, kx=3, ky=3)
Z_orig_smooth = interp_orig(y_fine, x_fine)

# 6. Graficar superficies suavizadas en escala BW
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_fine, Y_fine, Z_orig_smooth, cmap='gray', edgecolor='none')
ax1.set_title('Matriz Original (Grises)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Intensidad')

plt.tight_layout()
plt.show()
