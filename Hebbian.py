import numpy as np
import matplotlib.pyplot as plt

x = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
w = np.array([0.1,0.2])
ya = np.array([1,1,1,-1]) # NAND output
# ya = np.array([1,-1,-1,-1]) # NOR output
b = 0
n = 0.1

epochs = 0
while True:
    epochs+=1
    flag = True
    print("Epoch",epochs)
    for i in range(len(x)):
        w = w + n * ya[i]*x[i] 
        b = round(b + n * ya[i],2)
        print("Wnew for",x[i],":",w)
        print("bnew",x[i],":",b)
    for i in range(len(x)):
        net = round(np.dot(x[i],w)+b,2)
        yp = 1 if net>=0 else -1  
        print("Input",i+1,":",x[i],"  ","net:",net,"  ","Ya:",ya[i],"  ","Yp:",yp)
        if ya[i]!=yp:
            flag = False
    if flag:
        break
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
plt.title('Hebbian Perceptron Decision Boundary')
plt.grid()
plt.show()
