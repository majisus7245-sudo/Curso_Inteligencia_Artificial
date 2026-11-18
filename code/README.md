# Código de las tareas

| # | Carpeta | Tarea | Estado |
|---|---|---|---|
| 4 | [`04_regresion_lineal/`](04_regresion_lineal/) | Regresión lineal con California Housing | ✅ Lista |
| 5 | [`05_perceptron/`](05_perceptron/) | Ejemplo de perceptrón | ✅ Lista |
| 6 | [`06_dataset_rostros/`](06_dataset_rostros/) | Dataset propio de rostros | ⏳ Pendiente |
| 7 | [`07_clasificacion_digitos/`](07_clasificacion_digitos/) | Clasificación de dígitos 0-9 (MNIST) | ✅ Lista |
| 8 | [`08_cnn_rostros/`](08_cnn_rostros/) | CNN para clasificar rostros en tiempo real | ⏳ Pendiente |
| – | [`practicas_clase/`](practicas_clase/) | Ejercicios vistos en clase (MLP, CNN CIFAR-10) | – |

Los ejemplos de clase que van ligados a una tarea están junto a ella: el de los atletas
en `04_regresion_lineal/` y el de °C→°F en `05_perceptron/`.

Las tareas 1 (Árbol binario) y 3 (Búsquedas no informadas / 8-puzzle) están en
`Diagnostico/` y `MI_8PUZZLE/` en la raíz del repositorio.

Cada script se ejecuta **desde su propia carpeta**, por ejemplo:

```powershell
cd code\07_clasificacion_digitos
..\..\.venv\Scripts\python.exe dibujarnumero.py
```
