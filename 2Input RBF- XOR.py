import numpy as np
import matplotlib.pyplot as plt

# --- Data and Weights (Assume model is already trained/computed) ---
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
sd = 0.5 # spread
c = np.array([[0,0],[0,1],[1,0],[1,1]]) # Centers
# These weights 'w' must be calculated by running the full model code first.
# For demonstration, let's re-run the calculation portion:
l = 0.01 
Q = np.zeros((len(x), len(c)))
for i in range(len(x)):
    for j in range(len(c)):
        Q[i, j] = np.exp(-np.linalg.norm(x[i] - c[j])**2 / (2 * sd**2))
I = np.identity(Q.shape[1])
w = np.linalg.inv(Q.T @ Q + l * I) @ Q.T @ y

p = (Q @ w) 
print("RBF Activations (Q):\n", Q)
print("\nWeights (W):\n", w)
print("\nPredicted output:\n", p)

g = np.linspace(-0.5,1.5,200)
X1,X2 = np.meshgrid(g,g)
Z = np.array([np.exp(-np.linalg.norm(c-[i,j],axis=1)**2/(2*sd**2))@w 
              for i,j in zip(X1.ravel(),X2.ravel())]).reshape(X1.shape)

# --- Plot ---
plt.figure(figsize=(3,3))
plt.contour(X1,X2,Z,levels=[0.5],colors='k',linewidths=2,linestyles='--')
for xi, yi, zi in zip(x[:,0],x[:,1],y.ravel()):
    plt.scatter(xi, yi, s=100, marker='^' if zi else 'o', edgecolors='k', linewidths=1.5)
plt.title('RBF XOR Boundary ($σ=0.5$)', fontsize=12)
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.xlim(-0.1,1.1); plt.ylim(-0.1,1.1)
plt.xticks([0,1]); plt.yticks([0,1]); plt.grid(ls=':',alpha=0.7)
plt.tight_layout(); plt.show()
