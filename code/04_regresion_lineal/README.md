# Tarea 4 - Regresión Lineal (California Housing)

**Tarea:** ejercicio de regresión lineal usando el dataset California Housing
(ingreso medio `MedInc` → valor de la casa, primeras 50 zonas). El objetivo es ENTENDER
al 100% qué se está haciendo.

| Archivo | Qué es |
|---|---|
| `tarea-regresion-california.py` | **Entrega.** El mismo algoritmo del ejemplo de clase, ahora con California Housing (50 zonas), comentado paso a paso. |
| `lineal-regression.py` | Ejemplo de clase: regresión "from scratch" con datos simulados de atletas. |
| `extra-regresion-california-completa.py` | Extra (no lo pidieron): las 20,640 zonas, regresión múltiple con las 8 variables y comparación con scikit-learn. |

Requiere: `numpy`, `matplotlib`, `pandas`, `scikit-learn`. La primera ejecución descarga el dataset.

## Resultado

```
Precio = 0.4417 * (Ingreso Medio) + 0.6009
Solución exacta (mínimos cuadrados): w = 0.4416, b = 0.6012
```

→ Por cada **$10,000** más de ingreso medio en la zona, la casa vale en promedio **~$44,000 más**.

## Explicación en 4 pasos (se repiten en cada época)

1. **Predecir:** `Y_pred = w*X + b` con la recta actual (al inicio, `w = b = 0`).
2. **Medir el error (costo):** `MSE = (1/2n) Σ (Y_pred − Y)²`. Mientras más bajo, mejor la recta.
3. **Gradientes:** `dw` y `db` dicen hacia dónde crece el error si muevo `w` o `b`.
4. **Corregir:** `w = w − lr·dw`, `b = b − lr·db` → un paso en contra del error.

Después de 3000 épocas el costo deja de bajar (se aplana): el modelo **convergió**, y `w`, `b`
coinciden con la solución exacta de mínimos cuadrados.

**Unidades:** `MedInc` está en decenas de miles de USD (3.5 = $35,000) y `MedHouseVal`
en cientos de miles de USD (2.5 = $250,000).
