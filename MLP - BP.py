import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0,0],[0,1],[1,0],[1,1]])
t = np.array([0,1,1,0])   # XOR outputs
# t = np.array([1,0,0,1])   # XNOR outputs
w = np.array([[0.1,0.2],[-0.1,0.1]])
v = np.array([0.1,-0.2])    # weights hidden->output
bh = np.zeros(2)   # biases for hidden layer
bo = 0.0           # bias for output
n = 0.5            # learning rate
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sd(z):
    s = sigmoid(z)
    return s*(1-s)
epochs = 10000
for ep in range(epochs):
    total_error = 0
    for i in range(4):
        zin = np.dot(x[i], w.T) + bh      # input to hidden
        h = sigmoid(zin)                  # hidden layer output
        yin = np.dot(h, v) + bo           # input to output
        y = sigmoid(yin)                  # output
        e = t[i] - y
        total_error += 0.5*e**2
        do = e * sd(yin)                  # delta for output
        dh = do * v * sd(zin)             # delta for hidden neurons
        v += n * do * h
        bo += n * do
        w += n * np.outer(dh, x[i])       # update w (2x2)
        bh += n * dh

    if ep % 1000 == 0:
        print(f"Epoch {ep}, Error {total_error:.4f}")
# Final predictions
print("\nFinal predictions:")
for i in range(4):
    h = sigmoid(np.dot(x[i], w.T) + bh)
    y = sigmoid(np.dot(h, v) + bo)
    print(f"Input: {x[i]} -> Output: {y:.4f}")
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = []
for p in grid:
    h = sigmoid(np.dot(p, w.T) + bh)
    y = sigmoid(np.dot(h, v) + bo)
    Z.append(y)
Z = np.array(Z).reshape(xx.shape)
plt.figure(figsize=(3,3))
# Use plt.contour with levels=[0.5] to draw ONLY the boundary line
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='-', linewidths=2) 
plt.scatter(x[:,0], x[:,1], c=t, s=100, edgecolors='black', cmap='bwr', marker='o')
plt.title("Decision Boundary for XOR Neural Network")
plt.xlabel("Input x1")
plt.ylabel("Input x2")
plt.grid(True)
plt.show() 
