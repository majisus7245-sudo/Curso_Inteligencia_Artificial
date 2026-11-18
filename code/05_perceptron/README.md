# Tarea 5 - Ejemplo de Perceptrón (°C → °F)

Una sola neurona lineal aprende la fórmula `F = 1.8·C + 32` solo a partir de ejemplos.

| Archivo | Qué es |
|---|---|
| `perceptron.py` | **Ejemplo visto en clase.** Neurona lineal con `SGDRegressor` de scikit-learn: escala los datos, entrena, predice 15/25/37 °C y reconstruye W ≈ 1.79 y b ≈ 31.90 a la escala original. |
| `perceptron-celsius-fahrenheit.py` | Misma idea, con una capa `Dense(1)` de Keras (notebook del repo del profesor `chemarios/AI-Course`). |

Requiere: `numpy`, `scikit-learn` (`perceptron.py`) / `tensorflow` (versión Keras).
