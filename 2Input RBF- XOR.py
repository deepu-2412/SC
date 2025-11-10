import numpy as np
import matplotlib.pyplot as plt

# Data for XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Parameters
sd = 0.5       # spread
l = 0.01       # regularization parameter
b = 0.1        # bias
c = np.array([[0,0],[0,1],[1,0],[1,1]])

# Compute Gaussian RBF activations (Q matrix)
Q = np.zeros((len(x), len(c)))
for i in range(len(x)):
    for j in range(len(c)):
        Q[i, j] = np.exp(-np.linalg.norm(x[i] - c[j])**2 / (2 * sd**2))
# Compute weights using regularized least squares
I = np.identity(Q.shape[1])
w = np.linalg.inv(Q.T @ Q + l * I) @ Q.T @ y

p = (Q @ w) 

# Display numerical results
print("RBF Activations (Q):\n", Q)
print("\nWeights (W):\n", w)
print("\nPredicted output:\n", p)

# --- Generate grid for visualization ---
x1 = x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        phi = np.exp(-np.linalg.norm(c - [X1[i,j], X2[i,j]], axis=1)**2 / (2*sd**2))
        Z[i,j] = np.dot(phi, w)

# --- 3D Plot ---
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='k', alpha=0.85, linewidth=0.3)
ax.contour(X1, X2, Z, zdir='z', offset=-0.2, cmap='viridis', alpha=0.6)

# Color-coded training points (0=blue, 1=red)
for xi, yi, zi in zip(x[:,0], x[:,1], y.ravel()):
    color, marker = ('red','^') if zi == 1 else ('blue','o')
    ax.scatter(xi, yi, zi, color=color, s=70, marker=marker, edgecolors='k', zorder=10)

# --- Styling ---
ax.set_title('RBF Network Output Surface for XOR', fontsize=13, pad=20)
ax.set_xlabel('x₁', labelpad=10)
ax.set_ylabel('x₂', labelpad=10)
ax.set_zlabel('Predicted Output', labelpad=12)
ax.set_zlim(-0.3, 1.3)
ax.view_init(elev=25, azim=130)
fig.colorbar(surf, shrink=0.6, aspect=10, label='Output Value')
plt.tight_layout()
plt.show()
