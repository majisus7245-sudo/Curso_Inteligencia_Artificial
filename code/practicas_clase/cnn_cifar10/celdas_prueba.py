# Celdas para añadir al final del tutorial de CNN (CIFAR-10) en Colab:
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb
#
# Se asume que ya corriste el tutorial, así que existen:
#   - model        (Sequential, última capa Dense(10) SIN softmax: entrega logits)
#   - class_names  (['airplane', 'automobile', 'bird', 'cat', 'deer',
#                    'dog', 'frog', 'horse', 'ship', 'truck'])
#   - import tensorflow as tf
#   - import matplotlib.pyplot as plt
#
# Las imágenes de prueba (gato, perro, avión, auto, barco, caballo)
# están en imagenes_prueba.zip. Descomprime el zip en tu computadora
# y súbelas con la Celda 1, o descarga una por URL con la Celda 2.
#
# Importante para comentar en clase:
#   - El tutorial reporta ~70% de exactitud en el conjunto de prueba,
#     así que es normal que algunas fotos nuevas se clasifiquen mal.
#   - El modelo ve la foto en 32×32 píxeles; al reducir una foto real
#     se pierde mucho detalle.

# %% Celda 1
# Opción A: subir imágenes desde la computadora (puedes elegir varias a la vez).
from google.colab import files

uploaded = files.upload()  # diccionario: nombre_de_archivo -> bytes

print("Archivos subidos:", list(uploaded.keys()))


# %% Celda 2
# Opción B (alternativa): descargar una imagen desde una URL.
# Wikimedia a veces rechaza peticiones sin User-Agent; por eso lo configuramos.
import urllib.request

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "CNN-clase/1.0 (ejercicio universitario)")]
urllib.request.install_opener(opener)

# Ejemplo: el labrador de las imágenes de prueba. Puedes cambiar la URL.
url_imagen = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)
ruta_url = tf.keras.utils.get_file(origin=url_imagen, fname="perro_ejemplo.jpg")
print("Imagen descargada en:", ruta_url)


# %% Celda 3
# Preprocesamiento, igual que en el tutorial:
# abrir con PIL, convertir a RGB, redimensionar a 32×32, dividir entre 255
# y agregar la dimensión de lote (el modelo espera forma (lote, 32, 32, 3)).
from io import BytesIO
from PIL import Image
import numpy as np


def abrir_imagen(ruta_o_bytes):
    """Acepta una ruta (str) o los bytes que devuelve files.upload()."""
    if isinstance(ruta_o_bytes, (bytes, bytearray)):
        img = Image.open(BytesIO(ruta_o_bytes))
    else:
        img = Image.open(ruta_o_bytes)
    return img.convert("RGB")


def preprocesar(ruta_o_bytes, tamaño=(32, 32)):
    """Devuelve (lote, imagen_32, imagen_original).

    lote tiene forma (1, 32, 32, 3) y valores entre 0 y 1.
    """
    original = abrir_imagen(ruta_o_bytes)
    imagen_32 = original.resize(tamaño)  # PIL usa un filtro de alta calidad
    lote = np.array(imagen_32, dtype="float32") / 255.0
    lote = np.expand_dims(lote, axis=0)
    return lote, imagen_32, original


# %% Celda 4
# Predicción y visualización.
# El modelo entrega logits (números sin normalizar), no probabilidades.
# Por eso aplicamos tf.nn.softmax antes de interpretar el resultado.


def predecir_y_mostrar(ruta_o_bytes, titulo=None):
    lote, imagen_32, original = preprocesar(ruta_o_bytes)

    logits = model.predict(lote, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    indice = int(np.argmax(probs))
    clase = class_names[indice]
    confianza = float(probs[indice])

    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))

    ejes[0].imshow(original)
    ejes[0].set_title(titulo or "Imagen original")
    ejes[0].axis("off")

    # interpolation='nearest' muestra los píxeles grandes, como los ve la red
    ejes[1].imshow(imagen_32, interpolation="nearest")
    ejes[1].set_title(f"Así la ve el modelo (32×32)\n{clase}  ({confianza:.1%})")
    ejes[1].axis("off")

    ejes[2].bar(class_names, probs)
    ejes[2].set_ylim(0, 1)
    ejes[2].set_ylabel("Probabilidad")
    ejes[2].set_title("Probabilidades de las 10 clases")
    ejes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    print(f"Predicción: {clase}  (confianza: {confianza:.1%})")
    return clase, confianza, probs


# %% Celda 5
# Recorremos TODAS las imágenes subidas en la Celda 1.
for nombre, datos in uploaded.items():
    print("=" * 60)
    print("Archivo:", nombre)
    predecir_y_mostrar(datos, titulo=nombre)


# %% Celda 6
# Si usaste la Celda 2, descomenta la siguiente línea para probar
# la imagen descargada por URL:
# predecir_y_mostrar(ruta_url, titulo="Imagen desde URL")
