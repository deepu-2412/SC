# Z = XY+X+Y  on simplifying we get Z=X+Y
# Z = XY+X  ==> X(1+Y) ==> X
# Z = X+Y+X'Y  on simplifying we get Z=X+Y
# Z = XY+X'Y+XY'  on simplifying we get Z=X+Y

import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0,0],[0,1],[1,0,],[1,1,]])
z1,z2,z3,z4 =[],[],[],[]
def And(x,y): 
    return x and y
def Or(x,y): 
    return x or y
def Not(x): 
    return not x
for a,b in x:
    z1.append(int(Or(Or(And(a,b),a),b)))
    z2.append(int(Or(And(a,b),a)))
    z3.append(int(Or(Or(And(Not(a),b),a),b)))
    z4.append(int(Or(Or(And(a,b),And(Not(a),b)),And(a,Not(b))))) 

z1,z2,z3,z4 = np.array(z1),np.array(z2),np.array(z3),np.array(z4)
print("z1 :",z1)
print("z2 :",z2)
print("z3 :",z3)
print("z4 :",z4)

def training_testing(x,ya,w,b,n,label=""):
    print(f"Training of {label}")
    ep = 0
    while True:
        ep+=1
        print("Epoch:",ep," W :",w," ","b :",b)
        net = np.dot(x,w)+b
        yp = np.where(net>=0,1,0)
        e = ya - yp
        w = w + n * np.dot(e, x) 
        b = round(b + n *sum(e),2)
        print("Error :",sum(e!=0))
        if np.all(e == 0):
            break
    
    test_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    print(f"\nTesting the perceptron model for {label}:")
    for inp in test_inputs:
        net = np.dot(inp, w) + b
        pred = 1 if net >= 0 else 0
        print("Input:" ,inp,"Predicted: ",pred)
    plt.figure(figsize=(4, 4))
    for i in range(len(x)):
        color = 'blue' if ya[i] == 1 else 'red'
        marker = 'o' if ya[i] == 1 else 'x'
        plt.scatter(x[i][0], x[i][1], color=color, marker=marker)

    # Decision boundary: w1*x + w2*y + b = 0
    x_vals = np.linspace(-2, 2, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'g')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Perceptron Decision Boundary of {label}',)
    plt.grid()
    plt.show()    

training_testing(x,z1,np.array([0.4,0.1]),0.1,0.3,label="Z1")
training_testing(x,z2,np.array([0.3,0.2]),0,0.1,label="Z2")
training_testing(x,z3,np.array([0.4,0.1]),0.1,0.3,label="Z3")
training_testing(x,z4,np.array([-0.1,-0.1]),0,0.6,label="Z4")
