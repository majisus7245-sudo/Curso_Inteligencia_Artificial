# Prácticas vistas en clase

No son entregas de la lista de tareas; son ejercicios de clase.

| Archivo / carpeta | Descripción |
|---|---|
| `MLP.py` | Perceptrón multicapa (MLP) con MNIST. |
| `cnn_cifar10/` | Práctica de CNN con CIFAR-10 en Google Colab (ver abajo). |

## `cnn_cifar10/` - Probar la CNN de CIFAR-10 con fotos reales

1. En Colab se corre el tutorial oficial de TensorFlow
   (https://www.tensorflow.org/tutorials/images/cnn), que entrena una CNN para reconocer
   10 clases: avión, auto, pájaro, gato, venado, perro, rana, caballo, barco y camión.
2. Al final se pegan las celdas de `celdas_prueba.py` (cada `# %% Celda N` es una celda):
   - **Celda 1:** subir fotos desde tu compu (las 6 `.jpg` de esta carpeta).
   - **Celda 2:** alternativa, descargar una foto por URL.
   - **Celda 3:** preprocesar la foto igual que el dataset: RGB, 32×32 píxeles, valores entre 0 y 1.
   - **Celda 4:** predecir. El modelo devuelve *logits*; se les aplica `softmax` para obtener
     probabilidades. Muestra la foto original, cómo la ve la red (32×32) y la barra de las 10 clases.
   - **Celdas 5-6:** prueban todas las fotos subidas (o la de la URL).
3. `fuentes.txt`: autor y licencia (Wikimedia Commons) de cada foto. Solo es la atribución.

Es normal que algunas fotos salgan mal: el modelo tiene ~70% de exactitud y a 32×32 se pierde
mucho detalle.
