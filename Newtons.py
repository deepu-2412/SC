import numpy as np
import sympy as sp

# Define symbols
x1, x2 = sp.symbols('x1 x2')
f = x1**3 + 2*x1*x2 - x1**2 * x2**2

# Compute gradient
df_x1 = sp.diff(f, x1)
df_x2 = sp.diff(f, x2)
print("differentiation with respect to x1",df_x1)
print("differentiation with respect to x2",df_x2)
# Initialize point
x = np.array([1.0, -1.0])  
a = 0.1 

for i in range(15):
    g = np.array([
        float(df_x1.subs({x1: x[0], x2: x[1]})),
        float(df_x2.subs({x1: x[0], x2: x[1]}))])    
    x = x - a * g   
    print("Iteration : ",i+1,"x = ",x)
    
    if np.linalg.norm(g) < 1e-7:
        print("Converged")
        break
