import numpy as np

# --- 1. Membership Functions (MFs) ---

def tri_mf(x, a, b, c):
    if a > b or b > c: return 0.0
    if x <= a or x >= c: return 0.0
    if a < x <= b: return (x - a) / (b - a)
    if b < x < c: return (c - x) / (c - b)
    return 0.0

# MFs for y_t-1 (Low, Medium, High)
def mu_low_y(y): return tri_mf(y, -1, -1, 0)
def mu_med_y(y): return tri_mf(y, -0.5, 0, 0.5)
def mu_high_y(y): return tri_mf(y, 0, 1, 1)

# MFs for tau_t (Early, Medium, Late)
def mu_early_tau(tau): return tri_mf(tau, 0, 0, 0.5)
def mu_med_tau(tau): return tri_mf(tau, 0.75, 0.5, 0.75)
def mu_late_tau(tau): return tri_mf(tau, 0.5, 1, 1)

# Helper functions to fetch membership values (handles composite sets like 'Low Medium')
def get_mu_y(y_val, y_set):
    if y_set == 'Low': return mu_low_y(y_val)
    if y_set == 'Low Medium': return max(mu_low_y(y_val), mu_med_y(y_val))
    if y_set == 'Medium': return mu_med_y(y_val)
    if y_set == 'High': return mu_high_y(y_val)
    return 0.0

def get_mu_tau(tau_val, tau_set):
    if tau_set == 'Early': return mu_early_tau(tau_val)
    if tau_set == 'Med Late': return max(mu_med_tau(tau_val), mu_late_tau(tau_val))
    if tau_set == 'Med': return mu_med_tau(tau_val)
    if tau_set == 'Late': return mu_late_tau(tau_val)
    return 0.0

# --- 2. Rule Base and Parameters ---

# R: (y_t-1_set, tau_t_set, Mamdani_C_i)
RULE_BASE = [
    ('Low', 'Early', 0.3), ('Low', 'Med Late', 0.0), ('Low Medium', 'Early', 0.3),
    ('Medium', 'Med', 0.0), ('Medium', 'Late', -0.3), ('High', 'Early', -0.3),
    ('High', 'Late', 0.3)
]

# Sugeno Parameters (a_i, b_i, c_i)
SUGENO_PARAMS = [
    (0.6, 0.4, 0.1), (0.8, -0.2, 0.05), (0.7, 0.2, 0.05),
    (0.9, 0.0, 0.0), (0.885, -0.1, 0.02), (0.5, -0.2, 0.2),
    (1.1, 0.5, -0.1)
]

# --- 3. Core Calculation Functions ---

def calculate_weights(y_prev, tau):
    """Calculate the rule firing strength (w_i = min(mu_Ai, mu_Bi))."""
    weights = []
    for y_set, tau_set, _ in RULE_BASE:
        mu_y = get_mu_y(y_prev, y_set)
        mu_tau = get_mu_tau(tau, tau_set)
        w_i = min(mu_y, mu_tau)
        weights.append(w_i)
    return weights

def mamdani_step(y_prev, tau, weights):
    """Predict y_t using Mamdani (Delta y = Sum w_i * C_i / Sum w_i)."""
    consequents = [c for _, _, c in RULE_BASE]
    sum_w_i_C_i = sum(w * c for w, c in zip(weights, consequents))
    sum_w_i = sum(weights)

    delta_y = sum_w_i_C_i / sum_w_i if sum_w_i != 0 else 0.0
    y_next = y_prev + delta_y
    return y_next, delta_y

def sugeno_step(y_prev, tau, weights):
    """Predict y_t using Sugeno (y_t = Sum w_i * f_i / Sum w_i)."""
    consequents_f = []
    for i in range(len(RULE_BASE)):
        a_i, b_i, c_i = SUGENO_PARAMS[i]
        f_i = a_i * y_prev + b_i * tau + c_i
        consequents_f.append(f_i)

    sum_w_i_f_i = sum(w * f for w, f in zip(weights, consequents_f))
    sum_w_i = sum(weights)

    y_next = sum_w_i_f_i / sum_w_i if sum_w_i != 0 else y_prev
    return y_next

# --- 4. Simulation Execution (T=1 to T=6) ---

# Inputs
T_norm = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Normalized Time T={1, 2, 3, 4, 5, 6}
Y_mamdani = [0.0] # Initial condition y0 = 0.0
Y_sugeno = [0.0]  # Initial condition y0 = 0.0

print("--- Fuzzy Time Series Simulation (T=1 to T=6) ---")
print("{:<5} {:<8} {:<8} {:<8} ".format("T", "tau_t", "M_y_t", "S_y_t"))
print("-------------------------------")

for t in range(1, 7):
    t_index = t - 1
    tau_t = T_norm[t_index]
    
    # Use the predicted value from the *previous* step for y_prev
    y_prev_mamdani = Y_mamdani[-1]
    y_prev_sugeno = Y_sugeno[-1]

    # Calculate Weights (Antecedents) - using Mamdani's y_prev for consistency in weights
    weights = calculate_weights(y_prev_mamdani, tau_t)

    # MAMADANI MODEL
    y_next_mamdani, delta_y = mamdani_step(y_prev_mamdani, tau_t, weights)
    Y_mamdani.append(y_next_mamdani)

    # SUGENO MODEL
    y_next_sugeno = sugeno_step(y_prev_sugeno, tau_t, weights)
    Y_sugeno.append(y_next_sugeno)

    # Print step results
    print("{:<5} {:<8.1f} {:<8.3f} {:<8.3f} ".format(
          t, tau_t, y_next_mamdani, y_next_sugeno))
import matplotlib.pyplot as plt
# Simulation results copied from your executed code
T_points = np.arange(0, 7) # Time points 0 to 6

plt.figure(figsize=(6, 5))
plt.plot(T_points, Y_mamdani, marker='o', linestyle='-', label='Mamdani Model', color='darkblue')
plt.plot(T_points, Y_sugeno, marker='s', linestyle='--', label='Sugeno Model', color='red')

# Annotate the T=1 to T=6 predictions
for t in range(1, 7):
    # Plot normalized time (tau_t) on the points for reference
    plt.annotate(f'$\\tau={T_norm[t-1]}$', 
                 (T_points[t] + 0.1, Y_mamdani[t] - 0.02), 
                 fontsize=8, color='darkblue', alpha=0.7)
    plt.annotate(f'$\\tau={T_norm[t-1]}$', 
                 (T_points[t] + 0.1, Y_sugeno[t] + 0.01), 
                 fontsize=8, color='red', alpha=0.7)

plt.title('Fuzzy Time Series Prediction (Mamdani vs. Sugeno)')
plt.xlabel('Time Step (T)')
plt.ylabel('Predicted Value ($y_t$)')
plt.xticks(T_points)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(-0.1, 1.0)
plt.show()
