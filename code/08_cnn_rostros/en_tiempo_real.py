"""
Tarea 8 - Reconocimiento de rostros en tiempo real.

Basado en el EnTiempoReal.py compartido por el profesor:
  1. Lee la webcam con OpenCV.
  2. DETECCIÓN PREVIA: un detector Haar Cascade encuentra los rostros y se recorta
     cada uno (la región de interés, ROI).
  3. INFERENCIA: cada rostro se pasa a la CNN entrenada (modelo_propia.keras o
     modelo_mobilenet.keras, según la variable MODELO).
  4. Se dibuja la caja (bounding box) con el nombre predicho y el % de confianza.
     Si la confianza es baja se muestra "Desconocido".

Requiere haber corrido antes entrenar_propia.py o entrenar_mobilenet.py.

Controles:  Q o ESC -> salir

Ejecutar desde esta carpeta:
   ..\\..\\.venv\\Scripts\\python.exe en_tiempo_real.py
"""

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
MODELO = "mobilenet"            # "propia" o "mobilenet": cuál modelo entrenado usar

AQUI = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_MODELO = os.path.join(AQUI, f"modelo_{MODELO}.keras")
ARCHIVO_CLASES = os.path.join(AQUI, f"clases_{MODELO}.json")

TAMANO_IMG = (128, 128)      # el mismo tamaño con el que se entrenó
CONFIANZA_MINIMA = 0.60      # por debajo de esto se muestra "Desconocido"
TAMANO_MIN_CARA = (80, 80)   # ignora detecciones muy pequeñas
CAMARA = 0                   # 0 = webcam integrada; prueba 1 si tienes una externa
ESPEJO = True                # voltear la imagen como espejo (igual que en la captura)

VERDE = (0, 200, 0)
ROJO = (0, 0, 230)


# ----------------------------------------------------------------------
# Carga del modelo, las clases y el detector
# ----------------------------------------------------------------------
def cargar_todo():
    for archivo in (ARCHIVO_MODELO, ARCHIVO_CLASES):
        if not os.path.exists(archivo):
            raise SystemExit(f"No existe {os.path.basename(archivo)}. Corre primero entrenar_{MODELO}.py")

    print("Cargando modelo...")
    modelo = tf.keras.models.load_model(ARCHIVO_MODELO)
    with open(ARCHIVO_CLASES, encoding="utf-8") as f:
        clases = json.load(f)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise SystemExit("No se pudo cargar el detector Haar (usa opencv-python 4.x).")
    return modelo, clases, detector


def preparar_rostro(frame_bgr, x, y, w, h):
    """Recorta el rostro y lo deja igual que las fotos de entrenamiento.

    - OpenCV entrega BGR, pero el modelo se entrenó con imágenes RGB -> se convierte.
    - Se redimensiona a 128x128.
    - NO se divide entre 255: la normalización va DENTRO del modelo (capa Rescaling),
      así que aquí se le pasan los píxeles tal cual (0-255).
    """
    rostro = frame_bgr[y:y + h, x:x + w]
    rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    rostro = cv2.resize(rostro, TAMANO_IMG, interpolation=cv2.INTER_AREA)
    return rostro.astype("float32")


def dibujar_etiqueta(frame, x, y, w, h, texto, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_texto = y - 10 if y - th - 14 > 0 else y + h + th + 10
    cv2.rectangle(frame, (x, y_texto - th - 6), (x + tw + 8, y_texto + base - 2), color, -1)
    cv2.putText(frame, texto, (x + 4, y_texto - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    modelo, clases, detector = cargar_todo()
    print(f"Clases: {clases}")

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. Revisa que no la esté usando otra app.")

    print("Reconocimiento en marcha. Presiona Q o ESC para salir.")
    anterior = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break
            if ESPEJO:
                frame = cv2.flip(frame, 1)

            # 1. Detección previa de rostros (ROI)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = detector.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5, minSize=TAMANO_MIN_CARA)

            if len(caras) > 0:
                # 2. Inferencia: todos los rostros del cuadro en un solo lote (más rápido)
                lote = np.stack([preparar_rostro(frame, x, y, w, h) for (x, y, w, h) in caras])
                probabilidades = modelo(lote, training=False).numpy()

                # 3. Inferencia visual: caja + etiqueta + confianza
                for (x, y, w, h), probs in zip(caras, probabilidades):
                    indice = int(np.argmax(probs))
                    confianza = float(probs[indice])
                    if confianza < CONFIANZA_MINIMA:
                        dibujar_etiqueta(frame, x, y, w, h, f"Desconocido ({confianza:.0%})", ROJO)
                    else:
                        dibujar_etiqueta(frame, x, y, w, h, f"{clases[indice]} ({confianza:.0%})", VERDE)

            # FPS (cuadros por segundo) suavizado
            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(ahora - anterior, 1e-6))
            anterior = ahora
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Reconocimiento Facial - CNN", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
