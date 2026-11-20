import os
import re
import unicodedata

import cv2

# Configuración
CARPETA_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TAMANO_ROSTRO = (128, 128)   # tamaño uniforme del dataset (requisito: 128x128)
GUARDAR_CADA = 4             # guarda 1 de cada N cuadros -> evita fotos casi idénticas
TAMANO_MIN_CARA = (100, 100) # ignora caras muy pequeñas (lejanas o falsas detecciones)
CAMARA = 0                   # 0 = webcam integrada; prueba 1 si tienes una externa

# Condiciones del menú: (clave para el archivo, descripción, fotos sugeridas)
CONDICIONES = [
    ("frente",       "De frente, cara neutral",                         50),
    ("angulo_izq",   "Girando la cabeza hacia la izquierda (~30°)",     40),
    ("angulo_der",   "Girando la cabeza hacia la derecha (~30°)",       40),
    ("arriba_abajo", "Inclinando la cabeza arriba y abajo",             40),
    ("gestos",       "Gestos: sonreír, hablar, serio, cejas",           40),
    ("luz_ventana",  "Luz natural (ventana, de día)",                   40),
    ("luz_foco",     "Luz artificial (foco, de noche)",                 40),
    ("poca_luz",     "Poca luz (pero que se vea la cara)",              30),
    ("lentes",       "Con lentes (solo si usas)",                       40),
    ("distancia",    "Acercándose y alejándose de la cámara",           30),
]

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if detector.empty():
    raise SystemExit("No se pudo cargar el detector de rostros (instala opencv-python 4.x).")


# Utilidades
def limpiar_nombre(texto):
    # 'José Pérez' -> 'jose_perez' 
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto.strip().lower())
    return texto.strip("_")


def pedir_texto(mensaje, por_defecto=None):
    while True:
        valor = limpiar_nombre(input(mensaje))
        if valor:
            return valor
        if por_defecto:
            return por_defecto
        print("  -> Escribe algo válido (letras o números).")


def fotos_por_condicion(carpeta, persona):
    # Cuenta cuántas fotos hay de cada condición, según el nombre del archivo.
    conteo = {}
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            m = re.match(rf"{re.escape(persona)}_(.+)_\d+\.(jpg|jpeg|png)$", archivo, re.IGNORECASE)
            if m:
                conteo[m.group(1)] = conteo.get(m.group(1), 0) + 1
    return conteo


def elegir_condicion(carpeta, persona):
    # Muestra el menú de condiciones y devuelve (clave, fotos_sugeridas).
    conteo = fotos_por_condicion(carpeta, persona)
    print("\n¿Qué condición vas a capturar?")
    print(f"   {'#':>2}  {'Condición':<46}{'Llevas':>7}{'Sugeridas':>11}")
    for i, (clave, descripcion, sugeridas) in enumerate(CONDICIONES, start=1):
        llevas = conteo.get(clave, 0)
        marca = "  ✓" if llevas >= sugeridas else ""
        print(f"   {i:>2}  {descripcion:<46}{llevas:>7}{sugeridas:>11}{marca}")
    otra = len(CONDICIONES) + 1
    print(f"   {otra:>2}  Otra (escribirla)")

    while True:
        opcion = input(f"Elige una opción (1-{otra}): ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(CONDICIONES):
            clave, descripcion, sugeridas = CONDICIONES[int(opcion) - 1]
            print(f"  -> {descripcion}")
            return clave, sugeridas
        if opcion == str(otra):
            return pedir_texto("  Escribe el nombre de la condición: "), 40
        print(f"  -> Escribe un número del 1 al {otra}.")


def pedir_entero(mensaje, por_defecto):
    while True:
        valor = input(mensaje).strip()
        if not valor:
            return por_defecto
        if valor.isdigit() and int(valor) > 0:
            return int(valor)
        print("  -> Escribe un número entero mayor que 0.")


def siguiente_numero(carpeta):
    # Busca el número más alto ya usado en la carpeta para continuar desde ahí.
    maximo = 0
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", archivo, re.IGNORECASE)
            if m:
                maximo = max(maximo, int(m.group(1)))
    return maximo + 1


def contar_fotos(carpeta):
    if not os.path.isdir(carpeta):
        return 0
    return sum(1 for a in os.listdir(carpeta) if a.lower().endswith((".jpg", ".jpeg", ".png")))


def texto(frame, msg, y, color=(255, 255, 255), escala=0.7):
    # Borde negro para que se lea sobre cualquier fondo (putText no admite acentos)
    cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


# Una sesión de captura
def capturar_sesion(cap, persona, condicion, total):
    carpeta = os.path.join(CARPETA_DATASET, persona)
    os.makedirs(carpeta, exist_ok=True)
    numero = siguiente_numero(carpeta)

    guardadas = 0
    cuadro = 0
    pausado = True  # empieza en pausa para que te acomodes
    ventana = "Captura de rostros"

    while guardadas < total:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la cámara.")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo: más natural para acomodarse
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = detector.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5, minSize=TAMANO_MIN_CARA)

        vista = frame.copy()
        if len(caras) > 0:
            # Solo la cara más grande: si alguien pasa atrás, no se mezcla en tu carpeta
            x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
            color = (0, 200, 255) if pausado else (0, 255, 0)
            cv2.rectangle(vista, (x, y), (x + w, y + h), color, 2)

            cuadro += 1
            if not pausado and cuadro % GUARDAR_CADA == 0:
                rostro = cv2.resize(frame[y:y + h, x:x + w], TAMANO_ROSTRO, interpolation=cv2.INTER_AREA)
                nombre = f"{persona}_{condicion}_{numero:04d}.jpg"
                cv2.imwrite(os.path.join(carpeta, nombre), rostro)
                numero += 1
                guardadas += 1
        elif not pausado:
            texto(vista, "No se detecta rostro", 110, (0, 0, 255))

        # Información en pantalla
        texto(vista, f"{persona} | {condicion} | {guardadas}/{total}", 30)
        if pausado:
            texto(vista, "PAUSA - ESPACIO para empezar / continuar", 70, (0, 200, 255))
        else:
            texto(vista, "Capturando... muevete despacio (ESPACIO = pausa)", 70, (0, 255, 0))
        alto = vista.shape[0]
        cv2.rectangle(vista, (0, alto - 12), (int(vista.shape[1] * guardadas / total), alto), (0, 255, 0), -1)

        cv2.imshow(ventana, vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            pausado = not pausado
        elif tecla in (ord("q"), 27):  # Q o ESC
            print("Sesión terminada antes de tiempo.")
            break

    cv2.destroyWindow(ventana)
    cv2.waitKey(1)
    return guardadas, carpeta


# Programa principal
def main():
    print("=" * 60)
    print("  CAPTURA DE ROSTROS PARA EL DATASET")
    print("=" * 60)

    persona = pedir_texto("Nombre del integrante: ")
    carpeta = os.path.join(CARPETA_DATASET, persona)
    ya = contar_fotos(carpeta)
    if ya:
        print(f"  '{persona}' ya tiene {ya} fotos. Las nuevas se agregan (no se sobrescribe nada).")

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. Revisa que no la esté usando otra app (Teams, Zoom...).")

    try:
        while True:
            condicion, sugeridas = elegir_condicion(carpeta, persona)
            total = pedir_entero(f"¿Cuántas fotos tomar? [Enter = {sugeridas}]: ", por_defecto=sugeridas)

            print("\nSe abrió la cámara EN PAUSA. Acomódate y presiona ESPACIO para empezar.")
            guardadas, carpeta = capturar_sesion(cap, persona, condicion, total)
            print(f"  -> Se guardaron {guardadas} fotos de '{condicion}'. "
                  f"Total de {persona}: {contar_fotos(carpeta)} fotos.")

            otra = input("\n¿Otra sesión para el mismo integrante? (s/n) [s]: ").strip().lower()
            if otra == "n":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nListo. Fotos en: {carpeta}")


if __name__ == "__main__":
    main()
