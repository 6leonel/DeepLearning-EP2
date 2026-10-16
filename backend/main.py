"""Evaluacion Parcial 2 - Deep Learning.

Dos servicios de prueba sin modificar el comportamiento de Android:

* POST /predecir        -> CNN original de 264 clases, matriz RGB 128x128x3.
* POST /predecir-yunet  -> CNN de 258 clases, fotografia JPEG/PNG en base64.

El modelo nuevo necesita que YuNet detecte y recorte la cara ANTES de
redimensionar a 128x128. El umbral 0.70 es PROVISIONAL, no calibrado.
"""

import base64
import binascii
import io
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel


# ==========================================
# 1. ARCHIVOS Y CONFIGURACION
# ==========================================

DIRECTORIO_BASE = Path(__file__).resolve().parent
DIRECTORIO_ANTIGUO = DIRECTORIO_BASE / "modelo"
DIRECTORIO_YUNET = DIRECTORIO_BASE / "modelo_yunet"

RUTA_MODELO_ANTIGUO = DIRECTORIO_ANTIGUO / "modelo_rostros_cnn_ep2.keras"
RUTA_CLASES_ANTIGUAS = DIRECTORIO_ANTIGUO / "clases_cnn_ep2.json"
RUTA_CONFIG_ANTIGUA = DIRECTORIO_ANTIGUO / "preprocesamiento_cnn_ep2.json"

RUTA_MODELO_YUNET = DIRECTORIO_YUNET / "modelo_rostros_cnn_yunet_etapa2.keras"
RUTA_CLASES_YUNET = DIRECTORIO_YUNET / "clases_cnn_yunet_ep2.json"
RUTA_CONFIG_YUNET = DIRECTORIO_YUNET / "preprocesamiento_cnn_yunet_ep2.json"
RUTA_DETECTOR_YUNET = DIRECTORIO_YUNET / "face_detection_yunet_2023mar.onnx"

ENTRADA_CNN = (128, 128, 3)
UMBRAL_PROVISIONAL = 0.70
MAX_BYTES_IMAGEN = 12 * 1024 * 1024
MAX_PIXELES_IMAGEN = 24_000_000


# ==========================================
# 2. ESQUEMAS DE SOLICITUD
# ==========================================

class SolicitudPrediccion(BaseModel):
    imagen: list


class SolicitudPrediccionYuNet(BaseModel):
    imagen_base64: str


# ==========================================
# 3. CARGAR ETIQUETAS Y MODELOS
# ==========================================

def cargar_clases(ruta):
    clases_leidas = json.loads(ruta.read_text(encoding="utf-8"))
    clases_leidas = sorted(clases_leidas, key=lambda x: int(x["indice"]))

    for posicion, clase in enumerate(clases_leidas):
        if int(clase["indice"]) != posicion:
            raise ValueError(f"Indices incorrectos en {ruta.name}")
        if not str(clase.get("nombre_visible", "")).strip():
            raise ValueError(f"Falta nombre_visible en {ruta.name}")

    return clases_leidas


def cargar_recursos():
    requeridos = (
        RUTA_MODELO_ANTIGUO,
        RUTA_CLASES_ANTIGUAS,
        RUTA_CONFIG_ANTIGUA,
        RUTA_MODELO_YUNET,
        RUTA_CLASES_YUNET,
        RUTA_CONFIG_YUNET,
        RUTA_DETECTOR_YUNET,
    )

    for ruta in requeridos:
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta un archivo: {ruta}")

    config_antigua = json.loads(RUTA_CONFIG_ANTIGUA.read_text(encoding="utf-8"))
    config_nueva = json.loads(RUTA_CONFIG_YUNET.read_text(encoding="utf-8"))

    if config_antigua.get("entrada_modelo") != list(ENTRADA_CNN):
        raise ValueError("Configuracion incompatible de la CNN original")

    if config_nueva.get("modelo", {}).get("entrada") != list(ENTRADA_CNN):
        raise ValueError("Configuracion incompatible de la CNN YuNet")

    clases_antiguas = cargar_clases(RUTA_CLASES_ANTIGUAS)
    clases_nuevas = cargar_clases(RUTA_CLASES_YUNET)

    if len(clases_nuevas) != 258:
        raise ValueError("Se esperaban 258 identidades en el modelo YuNet")

    print("Cargando CNN anterior...", flush=True)
    modelo_antiguo = tf.keras.models.load_model(str(RUTA_MODELO_ANTIGUO), compile=False)

    print("Cargando CNN mejorada...", flush=True)
    modelo_nuevo = tf.keras.models.load_model(str(RUTA_MODELO_YUNET), compile=False)

    for modelo, clases, descripcion in (
        (modelo_antiguo, clases_antiguas, "anterior"),
        (modelo_nuevo, clases_nuevas, "nueva"),
    ):
        if tuple(modelo.input_shape[1:]) != ENTRADA_CNN:
            raise ValueError(f"Dimensiones incorrectas en CNN {descripcion}")
        if int(modelo.output_shape[-1]) != len(clases):
            raise ValueError(f"Cantidad de clases incorrecta en CNN {descripcion}")

    print("Cargando detector facial YuNet...", flush=True)
    detector = cv2.FaceDetectorYN.create(
        str(RUTA_DETECTOR_YUNET),
        "",
        (320, 320),
        0.80,
        0.30,
        5000,
    )

    print(f"CNN anterior: {len(clases_antiguas)} clases", flush=True)
    print(f"CNN nueva: {len(clases_nuevas)} identidades", flush=True)
    print(f"YuNet listo. OpenCV: {cv2.__version__}", flush=True)
    return modelo_antiguo, clases_antiguas, modelo_nuevo, clases_nuevas, detector


modelo_anterior, clases_anteriores, modelo_yunet, clases_yunet, detector_yunet = cargar_recursos()

# YuNet modifica internamente el tamaño de entrada, por eso protegemos
# la deteccion si dos peticiones llegan simultaneamente.
cerrojo_detector = threading.Lock()


# ==========================================
# 4. API Y ESTADOS
# ==========================================

app = FastAPI(
    title="EP2 - Reconocimiento Facial CNN + YuNet",
    description="CNN original y CNN mejorada, disponibles por rutas separadas.",
    version="3.0.0",
)


@app.get("/estado")
def estado():
    """Ruta anterior, compatible con Android actual."""
    return {
        "estado": "funcionando",
        "modelo": "CNN MobileNetV2",
        "clases": len(clases_anteriores),
        "entrada_android": list(ENTRADA_CNN),
    }


@app.get("/estado-yunet")
def estado_yunet():
    return {
        "estado": "funcionando",
        "modelo": "CNN MobileNetV2 + YuNet",
        "clases": len(clases_yunet),
        "entrada_android": "imagen_base64 (JPEG o PNG original)",
        "entrada_cnn": list(ENTRADA_CNN),
        "umbral_provisional": UMBRAL_PROVISIONAL,
    }


# ==========================================
# 5. FUNCIONES COMPARTIDAS
# ==========================================

def clasificar(modelo, clases, matriz, etiqueta):
    # La normalizacion (RGB 0..255 -> -1..1) ya esta en cada CNN.
    lote = np.expand_dims(matriz.astype(np.float32), axis=0)
    probabilidades = np.asarray(modelo.predict(lote, verbose=0)[0])

    if probabilidades.shape != (len(clases),) or not np.isfinite(probabilidades).all():
        raise HTTPException(status_code=500, detail="Prediccion invalida del modelo")

    indices = np.argsort(probabilidades)[-5:][::-1]
    principal = int(indices[0])
    confianza = float(probabilidades[principal])
    nombre_probable = clases[principal]["nombre_visible"]
    aceptado = confianza >= UMBRAL_PROVISIONAL

    print("\n" + "=" * 52, flush=True)
    print(f"DIAGNOSTICO: {etiqueta}", flush=True)
    print(f"Matriz RGB: {matriz.shape}", flush=True)
    print("TOP 5:", flush=True)
    for posicion, indice in enumerate(indices, start=1):
        numero = int(indice)
        print(
            f"  {posicion}. {clases[numero]['nombre_visible']} "
            f"- {float(probabilidades[numero]) * 100:.2f}%",
            flush=True,
        )
    print(
        f"Prediccion: {nombre_probable} / {confianza * 100:.2f}% "
        f"/ {'UMBRAL SUPERADO' if aceptado else 'UMBRAL NO SUPERADO'}",
        flush=True,
    )
    print("=" * 52, flush=True)

    return {
        "en_base": bool(aceptado),
        "nombre": nombre_probable if aceptado else "No se pudo verificar la identidad",
        "confianza": round(confianza, 6),
    }


# ==========================================
# 6. RUTA ANTIGUA: MATRIZ 128x128x3
# ==========================================

@app.post("/predecir")
def predecir(solicitud: SolicitudPrediccion):
    """Se mantiene el contrato actual de Android: {\"imagen\": [[[R,G,B], ...]]}."""
    try:
        matriz = np.asarray(solicitud.imagen, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=422, detail="La imagen debe ser una matriz numerica RGB")

    if matriz.shape != ENTRADA_CNN:
        raise HTTPException(
            status_code=422,
            detail=f"Se esperaba RGB 128x128x3; recibido: {list(matriz.shape)}",
        )

    if not np.isfinite(matriz).all() or np.any(matriz < 0) or np.any(matriz > 255):
        raise HTTPException(status_code=422, detail="Valores RGB invalidos (0 a 255)")

    return clasificar(modelo_anterior, clases_anteriores, matriz, "CNN ANTERIOR")


# ==========================================
# 7. DECODIFICAR FOTO ORIGINAL (BASE64)
# ==========================================

def leer_fotografia_base64(texto):
    texto = texto.strip()
    if texto.startswith("data:"):
        if "," not in texto:
            raise HTTPException(status_code=422, detail="Data URI incorrecta")
        texto = texto.split(",", 1)[1]

    # Base64 normalmente ocupa un 33 % mas que el archivo original.
    if not texto or len(texto) > (MAX_BYTES_IMAGEN * 4 // 3 + 100):
        raise HTTPException(status_code=413, detail="Fotografia demasiado grande o vacia")

    try:
        contenido = base64.b64decode(texto, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="imagen_base64 no es Base64 valido")

    if len(contenido) > MAX_BYTES_IMAGEN:
        raise HTTPException(status_code=413, detail="Fotografia superior a 12 MB")

    try:
        with Image.open(io.BytesIO(contenido)) as original:
            if original.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(status_code=422, detail="Formato no admitido: usa JPEG o PNG")
            ancho, alto = original.size
            if ancho < 100 or alto < 100:
                raise HTTPException(status_code=422, detail="La fotografia es demasiado pequena")
            if ancho * alto > MAX_PIXELES_IMAGEN:
                raise HTTPException(status_code=413, detail="Imagen demasiado grande en pixeles")
            fotografia = ImageOps.exif_transpose(original).convert("RGB").copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=422, detail="No se pudo leer la fotografia")

    return fotografia


# ==========================================
# 8. DETECCION Y RECORTE IGUAL A COLAB
# ==========================================

def recortar_con_yunet(fotografia):
    rgb = np.asarray(fotografia)
    alto, ancho = rgb.shape[:2]

    escala = min(1.0, 640.0 / max(alto, ancho))
    ancho_nuevo = max(1, round(ancho * escala))
    alto_nuevo = max(1, round(alto * escala))

    reducida = cv2.resize(
        rgb,
        (ancho_nuevo, alto_nuevo),
        interpolation=cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR,
    )
    bgr = cv2.cvtColor(reducida, cv2.COLOR_RGB2BGR)

    with cerrojo_detector:
        detector_yunet.setInputSize((ancho_nuevo, alto_nuevo))
        _, rostros = detector_yunet.detect(bgr)

    if rostros is None or len(rostros) == 0:
        return None, "sin_rostro"
    if len(rostros) > 1:
        return None, "varios_rostros"

    x, y, w, h = (float(v) for v in rostros[0][:4])
    x /= escala
    y /= escala
    w /= escala
    h /= escala

    centro_x = x + w / 2
    centro_y = y + h / 2
    lado = min(max(1, round(max(w, h) * 1.40)), ancho, alto)

    izquierda = max(0, min(round(centro_x - lado / 2), ancho - lado))
    arriba = max(0, min(round(centro_y - lado / 2), alto - lado))

    recorte = fotografia.crop(
        (izquierda, arriba, izquierda + lado, arriba + lado)
    )
    recorte = recorte.resize((128, 128), Image.Resampling.LANCZOS)
    matriz = np.asarray(recorte.convert("RGB"), dtype=np.uint8)

    if matriz.shape != ENTRADA_CNN:
        raise HTTPException(status_code=500, detail="Error interno al recortar rostro")

    return matriz, "detectado"


# ==========================================
# 9. RUTA NUEVA: SELFIE ORIGINAL EN BASE64
# ==========================================

@app.post("/predecir-yunet")
def predecir_yunet(solicitud: SolicitudPrediccionYuNet):
    """JSON: {\"imagen_base64\": \"/9j/4AAQSkZJRg...\"}."""
    fotografia = leer_fotografia_base64(solicitud.imagen_base64)
    matriz, estado = recortar_con_yunet(fotografia)

    if matriz is None:
        print(f"YuNet no clasifico: {estado}", flush=True)
        return {
            "en_base": False,
            "nombre": "No se pudo verificar la identidad",
            "confianza": 0.0,
            "motivo": estado,
        }

    respuesta = clasificar(modelo_yunet, clases_yunet, matriz, "CNN NUEVA + YUNET")
    respuesta["motivo"] = "rostro_detectado_umbral_provisional"
    return respuesta
