# Tarea 7 - Clasificación de dígitos 0-9 (MNIST)

| Archivo | Descripción |
|---|---|
| `dibujarnumero.py` | **Entrega.** Entrena una CNN con MNIST (o carga la ya entrenada) y abre una ventana para dibujar un número y que el modelo lo prediga en tiempo real. |
| `dibujarnumero_analisis.py` | Igual que el anterior, más una animación de cómo analiza la red el dibujo capa por capa. |
| `modelo_mnist.keras` | Modelo ya entrenado y guardado. |

**Importante:** ejecútalo desde esta carpeta. Si no, no encuentra `modelo_mnist.keras` y vuelve a entrenar.

```powershell
cd code\07_clasificacion_digitos
..\..\.venv\Scripts\python.exe dibujarnumero.py
```

Requiere: `tensorflow`, `numpy`, `pillow` (y `matplotlib` para la versión con análisis).
