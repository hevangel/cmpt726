import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')


def y1(x1, x2):
    return 6 + 2*x1**2 + 2*x2**2

def y2(x1, x2):
    return 8

x1 = np.linspace(-6, 6, 30)
x2 = np.linspace(-6, 6, 30)

X1, X2 = np.meshgrid(x1, x2)
Y1 = y1(X1, X2)
Y2 = y2(X1, X2)

ax.contour3D(X1, X2, Y1, 50, cmap='binary')
ax.contour3D(X1, X2, Y2, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y');

plt.show()

