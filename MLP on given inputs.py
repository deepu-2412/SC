import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
y = np.array([[0], [0.25], [0.5], [0.75], [1]], dtype=float)
etas = [0.1, 0.01, 0.001]
max_epochs = 100000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sd(x):
    s = sigmoid(x)
    return s * (1 - s)

# Initial weights and biases
w1_init = np.array([[0.1, 0.2, -0.1, 0.4, -0.3]], dtype=float)
b1_init = np.array([[0.5, 1, -0.5, 0, -0.1]], dtype=float)
w2_init = np.array([[0.2], [-0.2], [-0.1], [0.4], [-0.3]], dtype=float)
b2_init = np.array([[0.5]], dtype=float)
o = np.ones((5, 1), dtype=float)

plt.figure(figsize=(7,5))

for eta in etas:
    # Reset weights and biases for each eta
    w1 = w1_init.copy()
    b1 = b1_init.copy()
    w2 = w2_init.copy()
    b2 = b2_init.copy()
    
    mse = []
    
    for epoch in range(max_epochs):
        # Forward propagation
        z1 = x @ w1 + o @ b1
        A1 = sigmoid(z1)
        z2 = A1 @ w2 + o @ b2
        A2 = sigmoid(z2)
        E = y - A2
        MSE = np.mean(E**2)
        mse.append(MSE)
        if MSE < 1e-6:
            break
        
        # Backpropagation
        d2 = E * sd(z2)
        dw2 = A1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)
        d1 = (d2 @ w2.T) * sd(z1)
        dw1 = x.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)
        
        # Update weights and biases
        w2 += eta * dw2
        b2 += eta * db2
    
        w1 += eta * dw1
        b1 += eta * db1
    
    # Plot MSE for this eta
    plt.plot(mse, label=f"eta={eta}")
    print(f"\nFinished eta={eta}, final MSE={mse[-1]:.8f}")    
    # Predictions print("\nInput vs Predicted Output:") 
    for i in range(len(x)): 
        print(f"x = {x[i,0]}, predicted y = {A2[i,0]:.8f}")

# Final plot settings
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss vs Epochs for Different Learning Rates")
plt.legend()
plt.grid(True)  
plt.show()
