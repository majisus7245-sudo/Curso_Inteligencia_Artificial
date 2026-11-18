"""
TAREA: Regresión lineal "from scratch" con el dataset California Housing.

Es EXACTAMENTE el mismo algoritmo del ejemplo de los atletas (lineal-regression.py);
lo único que cambia son los DATOS:

    Atletas:     X = horas de entrenamiento   ->  Y = rendimiento
    California:  X = ingreso medio de la zona ->  Y = valor medio de la casa

Idea general: queremos encontrar la recta  Y = w*X + b  que mejor pasa por los puntos.
  - w (peso / pendiente): cuánto sube el precio por cada unidad de ingreso.
  - b (sesgo / intercepto): el precio "base" cuando el ingreso es 0.
No calculamos w y b con una fórmula; el programa los APRENDE poco a poco
(descenso de gradiente), que es la misma idea con la que aprende una neurona.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # fetch_california_housing(as_frame=True) devuelve tablas de pandas
from sklearn.datasets import fetch_california_housing

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =====================================================================
# 1. CARGA DE DATOS (código dado por el profesor)
# =====================================================================
california = fetch_california_housing(as_frame=True)  # la 1a vez lo descarga de internet

# california.data   -> DataFrame con 20,640 filas (zonas de California) y 8 columnas
# california.target -> Serie con el valor medio de la casa de cada zona
#
# Tomamos el ingreso medio (MedInc) vs el valor de la casa (MedHouseVal).
# Solo usamos las primeras 50 zonas para que sea pequeño y fácil de graficar/entender.
X = california.data['MedInc'].values[:50]   # .values convierte de pandas a arreglo de numpy
Y = california.target.values[:50]

# OJO CON LAS UNIDADES:
#   MedInc      está en DECENAS DE MILES de dólares   (3.5  -> $35,000 al año)
#   MedHouseVal está en CIENTOS DE MILES de dólares   (2.5  -> $250,000)
print(f"Datos cargados con éxito: {len(X)} zonas analizadas.")
print("Primeras 5 zonas (Ingreso, Precio):")
for xi, yi in zip(X[:5], Y[:5]):
    print(f"   Ingreso = {xi:.4f} (${xi * 10000:,.0f})  ->  Precio = {yi:.3f} (${yi * 100000:,.0f})")

# =====================================================================
# 2. INICIALIZACIÓN DE PARÁMETROS
# =====================================================================
# Empezamos "sin saber nada": la recta inicial es Y = 0*X + 0 (una línea plana en 0).
w = 0.0
b = 0.0

# Hiperparámetros (los elegimos nosotros, el modelo NO los aprende):
#   learning_rate: qué tan grande es cada paso al corregir w y b.
#                  Muy grande -> se pasa del mínimo y diverge.  Muy chico -> tarda muchísimo.
#   epochs:        cuántas veces repasamos TODOS los datos corrigiendo w y b.
learning_rate = 0.005
epochs = 3000
n = len(X)

historial_costos = []

# =====================================================================
# 3. ENTRENAMIENTO (descenso de gradiente)
# =====================================================================
# En cada época se repiten 4 pasos: predecir -> medir el error -> calcular hacia dónde
# corregir -> corregir.
for epoch in range(epochs):
    # (a) PREDICCIÓN con la recta actual, para las 50 zonas a la vez
    Y_pred = w * X + b

    # (b) COSTO: qué tan mal estamos.  Error Cuadrático Medio (dividido entre 2).
    #     (Y_pred - Y) es el error de cada zona; se eleva al cuadrado para que
    #     los errores negativos no cancelen a los positivos y para castigar más los grandes.
    cost = (1 / (2 * n)) * np.sum((Y_pred - Y) ** 2)
    historial_costos.append(cost)

    # (c) GRADIENTES: derivadas parciales del costo respecto a w y b.
    #     Dicen hacia dónde CRECE el error si aumento w (o b) un poquito.
    #       dw > 0  -> si subo w el error sube   -> hay que BAJAR w
    #       dw < 0  -> si subo w el error baja   -> hay que SUBIR w
    #     Nota: la derivada exacta de (1/2n)*Σ(error²) es (1/n)*Σ(error*X).
    #     El código de clase usa 2/n; eso solo multiplica el paso por 2 (equivale a
    #     usar un learning_rate el doble de grande). El resultado final es el mismo.
    dw = (2 / n) * np.sum((Y_pred - Y) * X)
    db = (2 / n) * np.sum(Y_pred - Y)

    # (d) ACTUALIZACIÓN: dar un paso en sentido CONTRARIO al gradiente (por eso el "-").
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Mostramos algunas épocas para VER cómo va aprendiendo
    if epoch in (0, 1, 2, 10, 100, 500, 1000, 2000) or epoch == epochs - 1:
        print(f"Época {epoch:>4}: costo = {cost:.5f} | w = {w:.4f} | b = {b:.4f}")

print("\n¡Entrenamiento completado!")
print(f"Ecuación final aprendida: Precio = {w:.4f} * (Ingreso Medio) + {b:.4f}")

# =====================================================================
# 4. INTERPRETACIÓN Y COMPROBACIÓN
# =====================================================================
print("\n--- ¿Qué significa? ---")
print(f"w = {w:.4f}: por cada $10,000 más de ingreso medio en la zona,")
print(f"           la casa vale en promedio ${w * 100000:,.0f} más.")
print(f"b = {b:.4f}: precio 'base' según la recta (${b * 100000:,.0f}) si el ingreso fuera 0.")
print("           (Es solo el punto donde la recta corta el eje Y; no hay zonas con ingreso 0.)")

# Comprobación: la recta óptima también se puede calcular directo con mínimos cuadrados.
# Si el descenso de gradiente funcionó, debe dar (casi) lo mismo.
w_exacto, b_exacto = np.polyfit(X, Y, 1)
print(f"\nSolución exacta (mínimos cuadrados): w = {w_exacto:.4f}, b = {b_exacto:.4f}")
print(f"Diferencia con lo aprendido:         w: {abs(w - w_exacto):.5f}, b: {abs(b - b_exacto):.5f}")

# Ejemplo de predicción
ingreso_ejemplo = 5.0
print(f"\nEjemplo: una zona con ingreso medio de ${ingreso_ejemplo * 10000:,.0f}")
print(f"  precio predicho = {w:.4f} * {ingreso_ejemplo} + {b:.4f} = {w * ingreso_ejemplo + b:.3f}"
      f"  (${(w * ingreso_ejemplo + b) * 100000:,.0f})")

# =====================================================================
# 5. VISUALIZACIÓN DE RESULTADOS
# =====================================================================
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='royalblue', s=70, label='Datos reales (zonas de California)', zorder=5)

X_linea = np.linspace(min(X) - 0.5, max(X) + 0.5, 100)
Y_linea = w * X_linea + b
plt.plot(X_linea, Y_linea, color='crimson', linewidth=3, label=f'Modelo lineal: Y = {w:.2f}X + {b:.2f}')

plt.title('Regresión Lineal From Scratch: Ingreso Medio vs Valor de la Casa (California Housing)',
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Ingreso medio de la zona (decenas de miles de USD)', fontsize=11)
plt.ylabel('Valor medio de la casa (cientos de miles de USD)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# El costo debe BAJAR rápido al inicio y luego aplanarse: eso indica que ya convergió.
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), historial_costos, color='darkorange', linewidth=2)
plt.title('Evolución de la Función de Costo (Error Cuadrático Medio)', fontsize=11, fontweight='bold')
plt.xlabel('Épocas / Iteraciones', fontsize=10)
plt.ylabel('Costo (MSE / 2)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
