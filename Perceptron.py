import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0,0],[0,1],[1,0],[1,1]])
w = np.array([0.9,-0.3])
ya = np.array([0,1,1,1]) # OR Inputs 
# ya = np.array([0,0,0,1]) # AND Inputs
b = 0.1

n = 0.4

epochs = 50
h=[]
for ep in range(epochs):
    net = np.dot(x,w)+b
    yp = np.where(net>=0,1,0)
    e = ya - yp
    w = w + n * np.dot(e, x) 
    b = b + n *sum(e) 
    h.append([w.copy(), b, np.sum(e != 0)])
    if np.all(e == 0):
        break
print(h)
# Plotting
plt.figure(figsize=(5, 5))

for i in range(len(x)):
    color = 'blue' if ya[i] == 1 else 'red'
    marker = 'o' if ya[i] == 1 else 'x'
    plt.scatter(x[i][0], x[i][1], color=color, marker=marker)
# Decision boundary: w1*x + w2*y + b = 0
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'g', label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(' Perceptron Decision Boundary')
plt.grid()
plt.show()
