
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

# Buscar la fotografía guardada para la prueba
carpeta = Path("backend/pruebas")
extensiones = {".jpg", ".jpeg", ".png", ".webp"}

fotografias = [
    archivo for archivo in carpeta.iterdir()
    if archivo.suffix.lower() in extensiones
]

if len(fotografias) != 1:
    raise RuntimeError(
        "Debe haber exactamente una fotografía en backend/pruebas"
    )

foto = fotografias[0]

# Convertir la fotografía a una matriz RGB de 128x128x3
with Image.open(foto) as imagen:
    imagen = ImageOps.exif_transpose(imagen)
    imagen = imagen.convert("RGB")
    imagen = imagen.resize((128, 128))
    matriz = list(imagen.getdata())

matriz = [
    [list(pixel) for pixel in matriz[fila * 128:(fila + 1) * 128]]
    for fila in range(128)
]

# Enviar la matriz a la API
datos = json.dumps({"imagen": matriz}).encode("utf-8")

solicitud = Request(
    "http://127.0.0.1:8000/predecir",
    data=datos,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(solicitud, timeout=60) as respuesta:
    resultado = json.loads(respuesta.read().decode("utf-8"))

    print("HTTP:", respuesta.status)
    print("Fotografía:", foto.name)
    print("Persona esperada: Grupo 08 - persona_27")
    print("Persona predicha:", resultado["nombre"])
    print("Confianza:", round(resultado["confianza"] * 100, 2), "%")
    print("Está en la base:", resultado["en_base"])
