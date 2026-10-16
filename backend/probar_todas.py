
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image, ImageOps
import numpy as np
import json

# ============================================
# CONFIGURACION
# ============================================

BASE = Path(__file__).resolve().parent
CARPETA_PRUEBAS = BASE / "pruebas"

URL_API = "http://127.0.0.1:8000/predecir"

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

# ============================================
# PROBAR UNA FOTOGRAFIA
# ============================================

def probar_imagen(ruta):

    with Image.open(ruta) as imagen:

        imagen = ImageOps.exif_transpose(imagen)
        imagen = imagen.convert("RGB")

        imagen = imagen.resize(
            (128, 128),
            Image.Resampling.BILINEAR
        )

        matriz = np.asarray(
            imagen,
            dtype=np.uint8
        ).tolist()

    datos = json.dumps({
        "imagen": matriz
    }).encode("utf-8")

    solicitud = Request(
        URL_API,
        data=datos,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(solicitud, timeout=60) as respuesta:
        return json.loads(
            respuesta.read().decode("utf-8")
        )


# ============================================
# EVALUAR TODAS LAS IMAGENES
# ============================================

resultados = []

for tipo in ["conocidos", "desconocidos"]:

    carpeta = CARPETA_PRUEBAS / tipo

    imagenes = sorted([
        archivo
        for archivo in carpeta.iterdir()
        if archivo.suffix.lower() in EXTENSIONES
    ])

    print("\n====================================")
    print("PROBANDO:", tipo.upper())
    print("====================================")

    for foto in imagenes:

        try:

            resultado = probar_imagen(foto)

            en_base = resultado["en_base"]
            nombre = resultado["nombre"]
            confianza = resultado["confianza"] * 100

            # Comprobar aceptacion o rechazo.
            # No verifica la identidad exacta.
            if tipo == "conocidos":
                correcto = en_base
            else:
                correcto = not en_base

            print("\nFotografia:", foto.name)
            print("Prediccion:", nombre)
            print(f"Confianza: {confianza:.2f}%")
            print("Registrado:", en_base)
            print("Aceptacion/rechazo correcto:", correcto)

            resultados.append({
                "tipo": tipo,
                "archivo": foto.name,
                "nombre": nombre,
                "confianza": confianza,
                "en_base": en_base,
                "aceptacion_correcta": correcto
            })

        except Exception as error:
            print("\nError en:", foto.name)
            print(error)


# ============================================
# RESUMEN FINAL
# ============================================

print("\n====================================")
print("RESUMEN DE LAS PRUEBAS")
print("====================================")

for tipo in ["conocidos", "desconocidos"]:

    grupo = [
        r for r in resultados
        if r["tipo"] == tipo
    ]

    correctos = sum(
        r["aceptacion_correcta"]
        for r in grupo
    )

    print(
        f"{tipo}: {correctos}/{len(grupo)} "
        "aceptados/rechazados correctamente"
    )

# Guardar los resultados
ruta_resultados = CARPETA_PRUEBAS / "resultados.json"

with open(
    ruta_resultados,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resultados,
        archivo,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:", ruta_resultados)
