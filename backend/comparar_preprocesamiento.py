
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Directorios del proyecto
BASE = Path(__file__).resolve().parent
MODELO_DIR = BASE / "modelo"
PRUEBAS_DIR = BASE / "pruebas"

# Cargar el modelo y sus clases
modelo = tf.keras.models.load_model(
    MODELO_DIR / "modelo_rostros_ep2.keras",
    compile=False
)

with open(MODELO_DIR / "clases_ep2.json", encoding="utf-8") as f:
    clases = json.load(f)

clases = sorted(clases, key=lambda c: c["indice"])

# Fotografia utilizada para la prueba
fotos = [
    p for p in PRUEBAS_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

if len(fotos) != 1:
    raise RuntimeError("Debe existir una sola fotografia en pruebas.")

with Image.open(fotos[0]) as original:
    original = ImageOps.exif_transpose(original)
    original.load()

# METODO 1: Igual que el entrenamiento en Colab
gris_directo = original.convert("L").resize(
    (64, 64),
    resample=Image.Resampling.BILINEAR
)

# METODO 2: Igual que la prueba actual de Android/API
rgb_android = original.convert("RGB").resize((128, 128))

gris_api = rgb_android.convert("L").resize(
    (64, 64),
    resample=Image.Resampling.BILINEAR
)

def evaluar(imagen, nombre_metodo):
    matriz = np.asarray(imagen, dtype=np.float32) / 255.0
    entrada = matriz.reshape(1, 64, 64, 1)

    probabilidades = modelo.predict(entrada, verbose=0)[0]
    indice = int(np.argmax(probabilidades))

    print(f"\n{nombre_metodo}")
    print("Persona predicha:", clases[indice]["nombre_visible"])
    print("Indice:", indice)
    print("Confianza maxima:", round(float(probabilidades[indice]) * 100, 2), "%")
    print("Confianza persona 200:", round(float(probabilidades[200]) * 100, 2), "%")

print("FOTOGRAFIA:", fotos[0].name)
print("PERSONA ESPERADA: Grupo 08 - persona_27")

evaluar(gris_directo, "METODO 1 - COLAB")
evaluar(gris_api, "METODO 2 - API")
