"""
Tarea 8 - Entrenamiento con TRANSFER LEARNING (MobileNetV2) para clasificar rostros.

Genera:  modelo_mobilenet.keras, clases_mobilenet.json, curvas_mobilenet.png

Ejecutar desde esta carpeta:
   ..\\..\\.venv\\Scripts\\python.exe entrenar_mobilenet.py
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(AQUI, "..", "06_dataset_rostros", "dataset")
TAMANO_IMG = (128, 128)
BATCH = 32
EPOCAS = 30          # máximo; EarlyStopping detiene antes si ya no mejora

# ----------------------------------------------------------------------
# 1. Dataset: 128x128, etiquetas one-hot, 80% entrenamiento / 20% validación
# ----------------------------------------------------------------------
entrenamiento, validacion = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=TAMANO_IMG,
    batch_size=BATCH,
    label_mode="categorical",
)
clases = entrenamiento.class_names
print(f"Clases ({len(clases)}): {clases}")

with open(os.path.join(AQUI, "clases_mobilenet.json"), "w", encoding="utf-8") as f:
    json.dump(clases, f, ensure_ascii=False, indent=2)

entrenamiento = entrenamiento.cache().shuffle(1000, seed=123).prefetch(tf.data.AUTOTUNE)
validacion = validacion.cache().prefetch(tf.data.AUTOTUNE)

# Balanceo: las clases con menos fotos pesan más -> peso = total / (n_clases * fotos_clase)
conteos = [len(os.listdir(os.path.join(DIR_DATASET, c))) for c in clases]
pesos_clase = {i: sum(conteos) / (len(clases) * n) for i, n in enumerate(conteos)}
for c, n, p in zip(clases, conteos, pesos_clase.values()):
    print(f"  {c:<16} {n:>4} fotos -> peso {p:.2f}")

# ----------------------------------------------------------------------
# 2. Arquitectura: MobileNetV2 preentrenada + salida adaptada a nuestras clases
# ----------------------------------------------------------------------
# Data Augmentation (solo actúa al entrenar)
aumento = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="aumento_de_datos")

base = tf.keras.applications.MobileNetV2(
    input_shape=TAMANO_IMG + (3,),
    include_top=False,       # sin la salida original de 1000 clases de ImageNet
    weights="imagenet",      # la primera vez descarga los pesos (~9 MB)
)
base.trainable = False       # congelada: solo se entrena la parte nueva

entrada = layers.Input(shape=TAMANO_IMG + (3,))
x = aumento(entrada)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # normalización a [-1, 1] (la que espera MobileNetV2)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
salida = layers.Dense(len(clases), activation="softmax")(x)   # una neurona por persona

modelo = models.Model(entrada, salida, name="mobilenetv2_rostros")
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
modelo.summary()

# ----------------------------------------------------------------------
# 3. Entrenamiento con callbacks
# ----------------------------------------------------------------------
historial = modelo.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=EPOCAS,
    class_weight=pesos_clase,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(os.path.join(AQUI, "modelo_mobilenet.keras"),
                                  monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ],
)

perdida, precision = modelo.evaluate(validacion, verbose=0)
print(f"\nMejor modelo -> val_loss: {perdida:.4f} | val_accuracy: {precision:.2%}")
print("Guardado en modelo_mobilenet.keras")

# Curvas de pérdida y accuracy
h = historial.history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(h["accuracy"], label="Entrenamiento")
ax1.plot(h["val_accuracy"], label="Validación")
ax1.set_title("Accuracy")
ax2.plot(h["loss"], label="Entrenamiento")
ax2.plot(h["val_loss"], label="Validación")
ax2.set_title("Pérdida (categorical_crossentropy)")
for ax in (ax1, ax2):
    ax.set_xlabel("Época")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
fig.suptitle("MobileNetV2 (Transfer Learning)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(AQUI, "curvas_mobilenet.png"), dpi=120)
plt.show()
