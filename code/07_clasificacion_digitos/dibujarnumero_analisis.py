"""
Reconocedor de dígitos dibujados a mano (con animación del análisis).

- Entrena (o carga) un modelo con MNIST.
- Abre una ventana donde dibujas un número con el mouse.
- "Predecir": el modelo dice qué número es.
- "Ver cómo analiza": abre una animación que muestra, capa por capa,
  lo que la red ve en tu dibujo.

Requisitos:  pip install tensorflow numpy pillow matplotlib
(tkinter ya viene incluido con Python en Windows/macOS;
 en Linux: sudo apt install python3-tk)
"""

import math
import os
import tkinter as tk

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

MODEL_PATH = "modelo_mnist.keras"


# ----------------------------------------------------------------------
# 1. Modelo: se entrena la primera vez y se guarda para las siguientes
# ----------------------------------------------------------------------
def obtener_modelo():
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo guardado...")
        return load_model(MODEL_PATH)

    print("Entrenando modelo (solo la primera vez)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    # CNN pequeña: reconoce mucho mejor dibujos hechos a mano que el MLP.
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
    model.save(MODEL_PATH)
    return model


# ----------------------------------------------------------------------
# 2. Preprocesado: deja el dibujo "al estilo MNIST"
#    (dígito en caja de 20x20, centrado por centro de masa en 28x28)
# ----------------------------------------------------------------------
def preparar_imagen(img_pil):
    """img_pil: imagen 'L' (escala de grises), fondo negro y trazo blanco."""
    arr = np.array(img_pil)
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return None  # lienzo vacío

    # Recortar al área del dígito
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape

    # Escalar para que el lado mayor mida 20 px (manteniendo proporción)
    escala = 20.0 / max(h, w)
    nuevo_w, nuevo_h = max(1, int(round(w * escala))), max(1, int(round(h * escala)))
    digito = Image.fromarray(arr).resize((nuevo_w, nuevo_h), Image.LANCZOS)

    # Pegarlo centrado en un lienzo de 28x28
    lienzo = Image.new("L", (28, 28), 0)
    lienzo.paste(digito, ((28 - nuevo_w) // 2, (28 - nuevo_h) // 2))

    # Mover para que el centro de masa quede en el centro (14, 14)
    a = np.array(lienzo).astype("float32")
    total = a.sum()
    cy, cx = (np.indices(a.shape) * a).sum(axis=(1, 2)) / total
    dx, dy = int(round(14 - cx)), int(round(14 - cy))
    a = np.roll(a, shift=(dy, dx), axis=(0, 1))

    return (a / 255.0).reshape(1, 28, 28, 1)


# ----------------------------------------------------------------------
# 3. Animación: muestra lo que "ve" cada capa de la red
# ----------------------------------------------------------------------
def obtener_activaciones(model, entrada):
    """Pasa la imagen capa por capa y guarda la salida de cada Conv2D y
    MaxPooling2D. También devuelve las probabilidades finales."""
    x = tf.constant(entrada)
    capas = {}
    conv_count, pool_count = 0, 0
    for layer in model.layers:
        x = layer(x, training=False)
        if isinstance(layer, Conv2D):
            conv_count += 1
            capas[f"conv{conv_count}"] = x.numpy()[0]
        elif isinstance(layer, MaxPooling2D):
            pool_count += 1
            capas[f"pool{pool_count}"] = x.numpy()[0]
    return capas, x.numpy()[0]


class VentanaAnimacion:
    PASO_ESCANEO = 12    # posiciones del filtro que avanza por cuadro
    INTERVALO_MS = 60    # velocidad de la animación

    def __init__(self, root, model, entrada):
        self.capas, self.probs = obtener_activaciones(model, entrada)
        self.entrada = entrada[0, :, :, 0]

        self.win = tk.Toplevel(root)
        self.win.title("Así analiza la red tu dibujo")
        self.fig = Figure(figsize=(9, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        barra = tk.Frame(self.win)
        barra.pack(pady=5)
        tk.Button(barra, text="Repetir", command=self.iniciar).pack(side="left", padx=5)
        tk.Button(barra, text="Cerrar", command=self.cerrar).pack(side="left", padx=5)
        self.win.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.cmap = matplotlib.colormaps["viridis"].copy()
        self.cmap.set_bad("#2b2b2b")  # zonas todavía sin calcular

        self.frames = self.construir_frames()
        self.anim = None
        self.iniciar()

    # ---- control ----
    def iniciar(self):
        if self.anim is not None and self.anim.event_source is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, lambda i: self.frames[i](),
                                  frames=len(self.frames),
                                  interval=self.INTERVALO_MS,
                                  repeat=False, blit=False)
        self.canvas.draw_idle()

    def cerrar(self):
        if self.anim is not None and self.anim.event_source is not None:
            self.anim.event_source.stop()
        self.win.destroy()

    # ---- construcción de los cuadros ----
    def construir_frames(self):
        frames = []
        conv1, pool1 = self.capas["conv1"], self.capas["pool1"]
        conv2, pool2 = self.capas["conv2"], self.capas["pool2"]

        # Etapa 1: un filtro 3x3 recorre la imagen (la "convolución")
        mejor = int(np.argmax(conv1.sum(axis=(0, 1))))  # filtro más activo
        mapa = conv1[:, :, mejor]
        total = mapa.size
        for p in range(self.PASO_ESCANEO, total + self.PASO_ESCANEO, self.PASO_ESCANEO):
            p = min(p, total)
            frames.append(lambda p=p: self.dibujar_escaneo(mapa, p, mejor))
        frames += [lambda: self.dibujar_escaneo(mapa, total, mejor)] * 12

        # Etapa 2 a 5: mapas de cada capa (aparecen poco a poco)
        etapas = [
            ("Paso 2: los 32 filtros de la 1.ª capa → 32 mapas de patrones simples\n"
             "(bordes, líneas, curvas). Brillante = patrón encontrado ahí.", conv1, 8),
            ("Paso 3: MaxPooling → mapas más pequeños (13×13)\n"
             "Se queda con lo más fuerte de cada zona.", pool1, 8),
            ("Paso 4: 2.ª capa convolucional → 64 mapas de formas más complejas\n"
             "(combinan los patrones simples: esquinas, lazos, cruces).", conv2, 8),
            ("Paso 5: MaxPooling otra vez → mapas de 5×5\n"
             "Resumen compacto de lo que hay en el dibujo.", pool2, 8),
        ]
        for titulo, mapas, cols in etapas:
            n = mapas.shape[-1]
            paso = max(1, n // 8)
            for k in range(paso, n + paso, paso):
                k = min(k, n)
                frames.append(lambda t=titulo, m=mapas, c=cols, k=k:
                              self.dibujar_cuadricula(t, m, c, k))
            frames += [lambda t=titulo, m=mapas, c=cols, n=n:
                       self.dibujar_cuadricula(t, m, c, n)] * 8

        # Etapa 6: resultado (las barras crecen)
        pasos = 12
        for k in range(1, pasos + 1):
            frames.append(lambda k=k: self.dibujar_resultado(k / pasos))
        frames += [lambda: self.dibujar_resultado(1.0)] * 20
        return frames

    # ---- dibujo de cada tipo de cuadro ----
    def dibujar_escaneo(self, mapa, p, filtro):
        fig = self.fig
        fig.clf()
        fig.suptitle("Paso 1: un filtro de 3×3 recorre la imagen\n"
                     f"(filtro #{filtro}) y anota qué tanto se parece cada zona a su patrón",
                     fontsize=11)
        ax1, ax2 = fig.subplots(1, 2)

        ax1.imshow(self.entrada, cmap="gray", vmin=0, vmax=1)
        pos = p - 1
        fila, col = divmod(pos, mapa.shape[1])
        ax1.add_patch(Rectangle((col - 0.5, fila - 0.5), 3, 3, fill=False,
                                edgecolor="red", linewidth=2))
        ax1.set_title("Imagen que ve la red (28×28)", fontsize=10)
        ax1.axis("off")

        parcial = np.full(mapa.shape, np.nan)
        parcial.flat[:p] = mapa.flat[:p]
        ax2.imshow(np.ma.masked_invalid(parcial), cmap=self.cmap,
                   vmin=mapa.min(), vmax=mapa.max())
        ax2.add_patch(Rectangle((col - 0.5, fila - 0.5), 1, 1, fill=False,
                                edgecolor="red", linewidth=1.5))
        ax2.set_title("Mapa que va construyendo (26×26)", fontsize=10)
        ax2.axis("off")
        fig.subplots_adjust(top=0.82, wspace=0.1)

    def dibujar_cuadricula(self, titulo, mapas, cols, n_mostrar):
        fig = self.fig
        fig.clf()
        fig.suptitle(titulo, fontsize=11)
        n = mapas.shape[-1]
        filas = math.ceil(n / cols)
        axes = fig.subplots(filas, cols)
        for i, ax in enumerate(np.atleast_1d(axes).flat):
            ax.axis("off")
            if i < n and i < n_mostrar:
                ax.imshow(mapas[:, :, i], cmap="viridis")
        fig.subplots_adjust(top=0.86, left=0.02, right=0.98,
                            bottom=0.02, wspace=0.08, hspace=0.08)

    def dibujar_resultado(self, avance):
        fig = self.fig
        fig.clf()
        mejor = int(np.argmax(self.probs))
        fig.suptitle(f"Resultado: la red concluye que es un {mejor} "
                     f"({self.probs[mejor] * 100:.1f}% de confianza)", fontsize=12)
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
        ax1.imshow(self.entrada, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Tu dibujo", fontsize=10)
        ax1.axis("off")

        colores = ["#1f77b4"] * 10
        colores[mejor] = "#d62728"
        ax2.bar(range(10), self.probs * avance, color=colores)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(10))
        ax2.set_xlabel("Dígito")
        ax2.set_ylabel("Probabilidad")
        ax2.set_title("Capas densas + softmax: una probabilidad por dígito",
                      fontsize=10)
        fig.subplots_adjust(top=0.82, wspace=0.25)


# ----------------------------------------------------------------------
# 4. Interfaz
# ----------------------------------------------------------------------
class App:
    TAM = 280      # tamaño del lienzo en pantalla
    GROSOR = 18    # grosor del pincel

    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Dibuja un número (0-9)")

        self.canvas = tk.Canvas(root, width=self.TAM, height=self.TAM,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Imagen paralela donde también se dibuja (la que ve el modelo)
        self.imagen = Image.new("L", (self.TAM, self.TAM), 0)
        self.draw = ImageDraw.Draw(self.imagen)

        self.canvas.bind("<B1-Motion>", self.pintar)
        self.canvas.bind("<ButtonRelease-1>", self.soltar)

        tk.Button(root, text="Predecir", command=self.predecir,
                  width=14).grid(row=1, column=0, pady=5)
        tk.Button(root, text="Borrar", command=self.borrar,
                  width=14).grid(row=1, column=1, pady=5)
        tk.Button(root, text="Ver cómo analiza", command=self.ver_animacion,
                  width=30).grid(row=2, column=0, columnspan=2, pady=5)

        self.resultado = tk.Label(root, text="Dibuja un número y presiona Predecir",
                                  font=("Arial", 14))
        self.resultado.grid(row=3, column=0, columnspan=2, pady=5)

        self.detalle = tk.Label(root, text="", font=("Arial", 10), fg="gray30")
        self.detalle.grid(row=4, column=0, columnspan=2, pady=(0, 10))

        self.ventana_anim = None

    def pintar(self, event):
        r = self.GROSOR // 2
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def soltar(self, _event):
        self.predecir()  # predice automáticamente al levantar el mouse

    def borrar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.TAM, self.TAM], fill=0)
        self.resultado.config(text="Dibuja un número y presiona Predecir")
        self.detalle.config(text="")

    def predecir(self):
        entrada = preparar_imagen(self.imagen)
        if entrada is None:
            self.resultado.config(text="El lienzo está vacío")
            return
        probs = self.model.predict(entrada, verbose=0)[0]
        orden = np.argsort(probs)[::-1]
        mejor = orden[0]
        self.resultado.config(
            text=f"Es un {mejor}  ({probs[mejor] * 100:.1f}% de confianza)")
        self.detalle.config(
            text=f"2.º: {orden[1]} ({probs[orden[1]] * 100:.1f}%)   "
                 f"3.º: {orden[2]} ({probs[orden[2]] * 100:.1f}%)")

    def ver_animacion(self):
        entrada = preparar_imagen(self.imagen)
        if entrada is None:
            self.resultado.config(text="Dibuja un número primero")
            return
        self.ventana_anim = VentanaAnimacion(self.root, self.model, entrada)


if __name__ == "__main__":
    modelo = obtener_modelo()
    ventana = tk.Tk()
    App(ventana, modelo)
    ventana.mainloop()