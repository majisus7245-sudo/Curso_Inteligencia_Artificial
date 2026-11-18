"""
Regresión Lineal con el dataset California Housing (costo de casas).

Se usa el mismo método visto en clase (descenso de gradiente "from scratch"),
pero ahora con datos reales:

  Parte A) Regresión lineal SIMPLE: ingreso medio de la zona -> precio de la casa
  Parte B) Regresión lineal MÚLTIPLE: las 8 características -> precio de la casa
  Parte C) Comparación contra LinearRegression de scikit-learn

Dataset: 20,640 zonas (block groups) de California, censo de 1990.
  Objetivo (Y): MedHouseVal = valor medio de la casa en cientos de miles de dólares.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =====================================================================
# 1. CARGA DE DATOS
# =====================================================================
housing = fetch_california_housing()  # la primera vez lo descarga de internet
X_todo = housing.data            # (20640, 8)
Y_todo = housing.target          # (20640,)
nombres = housing.feature_names  # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

print(f"Datos cargados con éxito: {X_todo.shape[0]} zonas, {X_todo.shape[1]} características.")
print("Características:", ", ".join(nombres))

# Separamos entrenamiento (80%) y prueba (20%) para medir qué tan bien generaliza
X_train, X_test, Y_train, Y_test = train_test_split(X_todo, Y_todo, test_size=0.2, random_state=42)

# Escalado (estandarización): media 0, desviación 1.
# Es necesario para que el descenso de gradiente converja con un learning rate razonable.
media = X_train.mean(axis=0)
desviacion = X_train.std(axis=0)
X_train_esc = (X_train - media) / desviacion
X_test_esc = (X_test - media) / desviacion


# =====================================================================
# PARTE A) REGRESIÓN LINEAL SIMPLE: MedInc -> Precio
# =====================================================================
print("\n========== PARTE A: Regresión simple (Ingreso medio -> Precio) ==========")

col = nombres.index("MedInc")
X = X_train[:, col]   # se usa sin escalar para que la ecuación sea fácil de interpretar
Y = Y_train
n = len(X)

w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 3000
historial_costos_simple = []

for epoch in range(epochs):
    Y_pred = w * X + b

    cost = (1 / (2 * n)) * np.sum((Y_pred - Y) ** 2)
    historial_costos_simple.append(cost)

    dw = (1 / n) * np.sum((Y_pred - Y) * X)
    db = (1 / n) * np.sum(Y_pred - Y)

    w = w - learning_rate * dw
    b = b - learning_rate * db

Y_test_pred_simple = w * X_test[:, col] + b
mse_simple = mean_squared_error(Y_test, Y_test_pred_simple)
r2_simple = r2_score(Y_test, Y_test_pred_simple)

print("¡Entrenamiento completado!")
print(f"Ecuación aprendida: Precio = {w:.4f} * (Ingreso Medio) + {b:.4f}")
print(f"Interpretación: por cada $10,000 más de ingreso medio, la casa vale ~${w * 100000:,.0f} más.")
print(f"Prueba -> MSE: {mse_simple:.4f} | R²: {r2_simple:.4f}")


# =====================================================================
# PARTE B) REGRESIÓN LINEAL MÚLTIPLE: 8 características -> Precio
#   Y_pred = X · W + b   (W es un vector con un peso por característica)
# =====================================================================
print("\n========== PARTE B: Regresión múltiple (8 características -> Precio) ==========")

Xm = X_train_esc
n, k = Xm.shape
W = np.zeros(k)
b_m = 0.0
learning_rate_m = 0.1
epochs_m = 2000
historial_costos_multi = []

for epoch in range(epochs_m):
    Y_pred = Xm @ W + b_m

    cost = (1 / (2 * n)) * np.sum((Y_pred - Y_train) ** 2)
    historial_costos_multi.append(cost)

    dW = (1 / n) * (Xm.T @ (Y_pred - Y_train))
    db = (1 / n) * np.sum(Y_pred - Y_train)

    W = W - learning_rate_m * dW
    b_m = b_m - learning_rate_m * db

Y_test_pred_multi = X_test_esc @ W + b_m
mse_multi = mean_squared_error(Y_test, Y_test_pred_multi)
r2_multi = r2_score(Y_test, Y_test_pred_multi)

print("¡Entrenamiento completado!")
print("Pesos aprendidos (sobre datos escalados):")
for nombre, peso in sorted(zip(nombres, W), key=lambda p: -abs(p[1])):
    print(f"  {nombre:<11} {peso:+.4f}")
print(f"  {'Sesgo (b)':<11} {b_m:+.4f}")
print(f"Prueba -> MSE: {mse_multi:.4f} | R²: {r2_multi:.4f}")


# =====================================================================
# PARTE C) COMPARACIÓN CON SCIKIT-LEARN
# =====================================================================
print("\n========== PARTE C: Comparación con scikit-learn ==========")

modelo_sk = LinearRegression()
modelo_sk.fit(X_train_esc, Y_train)
Y_test_pred_sk = modelo_sk.predict(X_test_esc)
mse_sk = mean_squared_error(Y_test, Y_test_pred_sk)
r2_sk = r2_score(Y_test, Y_test_pred_sk)

print(f"{'Modelo':<35}{'MSE':>10}{'R²':>10}")
print(f"{'Simple (MedInc) - from scratch':<35}{mse_simple:>10.4f}{r2_simple:>10.4f}")
print(f"{'Múltiple - from scratch':<35}{mse_multi:>10.4f}{r2_multi:>10.4f}")
print(f"{'Múltiple - scikit-learn':<35}{mse_sk:>10.4f}{r2_sk:>10.4f}")
print("Si el descenso de gradiente convergió, 'from scratch' y scikit-learn deben dar casi lo mismo.")


# =====================================================================
# 4. VISUALIZACIÓN DE RESULTADOS
# =====================================================================

# Gráfica 1: regresión simple
plt.figure(figsize=(10, 6))
idx = np.random.default_rng(0).choice(len(X_train), 3000, replace=False)  # muestra para no saturar
plt.scatter(X_train[idx, col], Y_train[idx], color='royalblue', s=10, alpha=0.4, label='Datos reales (zonas)')
X_linea = np.linspace(X_train[:, col].min(), X_train[:, col].max(), 100)
plt.plot(X_linea, w * X_linea + b, color='crimson', linewidth=3, label=f'Modelo: Y = {w:.2f}X + {b:.2f}')
plt.title('Regresión Lineal Simple: Ingreso Medio vs Precio de la Casa (California Housing)',
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Ingreso medio de la zona (decenas de miles de USD)', fontsize=11)
plt.ylabel('Valor medio de la casa (cientos de miles de USD)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Gráfica 2: evolución del costo
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), historial_costos_simple, color='darkorange', linewidth=2, label='Simple')
plt.plot(range(epochs_m), historial_costos_multi, color='seagreen', linewidth=2, label='Múltiple')
plt.title('Evolución de la Función de Costo (Error Cuadrático Medio)', fontsize=11, fontweight='bold')
plt.xlabel('Épocas / Iteraciones', fontsize=10)
plt.ylabel('Costo (MSE / 2)', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Gráfica 3: predicción vs valor real (regresión múltiple)
plt.figure(figsize=(7, 7))
plt.scatter(Y_test, Y_test_pred_multi, s=8, alpha=0.3, color='seagreen')
lim = [0, 5.2]
plt.plot(lim, lim, color='crimson', linestyle='--', linewidth=2, label='Predicción perfecta')
plt.xlim(lim)
plt.ylim(lim)
plt.title(f'Regresión Múltiple: Real vs Predicho (R² = {r2_multi:.3f})', fontsize=11, fontweight='bold')
plt.xlabel('Precio real (cientos de miles de USD)')
plt.ylabel('Precio predicho (cientos de miles de USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
