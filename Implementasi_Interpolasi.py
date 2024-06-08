import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, x_interp):
    def L(k, x_val):
        term = 1
        for i in range(len(x)):
            if i != k:
                term *= (x_val - x[i]) / (x[k] - x[i])
        return term
    
    y_interp = np.zeros_like(x_interp, dtype=float)
    for j in range(len(x_interp)):
        for i in range(len(x)):
            y_interp[j] += y[i] * L(i, x_interp[j])
    
    return y_interp

def newton_interpolation(x, y, x_interp):
    def divided_diff(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
        
        return coef[0, :]
    
    def newton_poly(coef, x_data, x_val):
        n = len(coef) - 1
        p = coef[n]
        for k in range(1, n + 1):
            p = coef[n - k] + (x_val - x_data[n - k]) * p
        return p
    
    coef = divided_diff(x, y)
    y_interp = np.array([newton_poly(coef, x, xi) for xi in x_interp])
    
    return y_interp

# Data
x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Interpolasi
x_interp = np.linspace(5, 40, 100)
y_interp_lagrange = lagrange_interpolation(x, y, x_interp)
y_interp_newton = newton_interpolation(x, y, x_interp)

# Plot kedua hasil interpolasi
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data asli')
plt.plot(x_interp, y_interp_lagrange, label='Interpolasi Lagrange')
plt.plot(x_interp, y_interp_newton, label='Interpolasi Newton', linestyle='--')
plt.xlabel('Tegangan (kg/mm²)')
plt.ylabel('Waktu patah (jam)')
plt.title('Interpolasi Lagrange vs Newton')
plt.legend()
plt.grid(True)
plt.show()


