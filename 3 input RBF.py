import numpy as np
import matplotlib.pyplot as plt

# --- Data and Parameters ---
x = np.array([
    [0,0,0], [0,0,1], [0,1,0], [0,1,1],
    [1,0,0], [1,0,1], [1,1,0], [1,1,1]
])
y = np.array([[0],[1],[1],[0],[1],[0],[0],[1]]) 

sd = 0.5         # spread (sigma)
l = 0.01         # regularization parameter (lambda)
c = x.copy()     # Centers are set to the data points
# --- 1. Compute Weights (W) ---
Q = np.zeros((len(x), len(c)))
for i in range(len(x)):
    for j in range(len(c)):
        Q[i, j] = np.exp(-np.linalg.norm(x[i] - c[j])**2 / (2 * sd**2))
I = np.identity(Q.shape[1])
w = np.linalg.inv(Q.T @ Q + l * I) @ Q.T @ y
p = (Q @ w) 

# --- Grid & Surface ---
x1, x2 = np.linspace(0,1,100), np.linspace(0,1,100)
X1, X2 = np.meshgrid(x1, x2)
x3 = 0.5
Z = np.array([
    np.exp(-np.linalg.norm(c - [i,j,x3], axis=1)**2/(2*sd**2))@w for i,j in zip(X1.ravel(), X2.ravel())
]).reshape(X1.shape)

# --- Plot ---
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.2)
for xi, yi, zi in zip(x[:,0], x[:,1], y.ravel()):
    ax.scatter(xi, yi, zi, c=('r' if zi else 'b'), s=60, edgecolors='k')
ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='Output', title='RBF Surface (x₃=0.5)')
ax.view_init(25,130); ax.set_zlim(-0.3,1.3)
plt.tight_layout(); plt.show()
