r"""Prueba local de la CNN con fotos conocidas y desconocidas.

Ejecutar desde la raiz del proyecto:
    & .\backend\.venv\Scripts\python.exe .\backend\probar_cnn.py

No modifica la API ni los modelos guardados.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


BASE = Path(__file__).resolve().parent
DIR_MODELO = BASE / "modelo"
DIR_PRUEBAS = BASE / "pruebas"
RUTA_MODELO = DIR_MODELO / "modelo_rostros_cnn_ep2.keras"
RUTA_CLASES = DIR_MODELO / "clases_cnn_ep2.json"
RUTA_CONFIG = DIR_MODELO / "preprocesamiento_cnn_ep2.json"
RUTA_RESULTADOS = DIR_PRUEBAS / "resultados_cnn.json"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}


def cargar_recursos():
    for ruta in (RUTA_MODELO, RUTA_CLASES, RUTA_CONFIG):
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontro: {ruta}")

    clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))
    clases = sorted(clases, key=lambda c: int(c["indice"]))
    config = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))

    if config.get("entrada_modelo") != [128, 128, 3]:
        raise ValueError("La configuracion CNN no indica entrada 128x128x3.")

    print("Cargando CNN. En CPU puede tardar un poco...")
    modelo = tf.keras.models.load_model(str(RUTA_MODELO), compile=False)

    if tuple(modelo.input_shape[1:]) != (128, 128, 3):
        raise ValueError(f"Entrada del modelo inesperada: {modelo.input_shape}")
    if int(modelo.output_shape[-1]) != len(clases):
        raise ValueError("El numero de salidas no coincide con clases_cnn_ep2.json")
    if any(int(clase["indice"]) != i for i, clase in enumerate(clases)):
        raise ValueError("Los indices del archivo de clases no son consecutivos.")

    print(f"Modelo cargado: {len(clases)} clases; entrada RGB 128x128x3")
    print("Normalizacion: dentro del modelo CNN (no dividir por 255).")
    return modelo, clases


def preparar_foto(ruta):
    with Image.open(ruta) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        imagen = imagen.resize((128, 128), Image.Resampling.BILINEAR)
        matriz = np.asarray(imagen, dtype=np.float32)
    return np.expand_dims(matriz, axis=0)


def ejecutar():
    modelo, clases = cargar_recursos()
    resultados = []

    for grupo in ("conocidos", "desconocidos"):
        carpeta = DIR_PRUEBAS / grupo
        if not carpeta.is_dir():
            raise FileNotFoundError(f"Falta la carpeta: {carpeta}")

        fotos = sorted(
            ruta for ruta in carpeta.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
        )

        print("\n" + "=" * 60)
        print(f"FOTOGRAFIAS: {grupo.upper()} ({len(fotos)})")
        print("=" * 60)

        for foto in fotos:
            try:
                matriz = preparar_foto(foto)
                probabilidades = np.asarray(
                    modelo.predict(matriz, verbose=0)[0], dtype=np.float64
                )
                mejores = np.argsort(probabilidades)[::-1][:3]
                predicciones = [
                    {
                        "indice": int(i),
                        "nombre": clases[int(i)]["nombre_visible"],
                        "confianza": round(float(probabilidades[i]), 6),
                    }
                    for i in mejores
                ]

                print(f"\nFoto: {foto.name}")
                for puesto, pred in enumerate(predicciones, start=1):
                    print(
                        f"  Top {puesto}: {pred['nombre']} "
                        f"({pred['confianza'] * 100:.2f}%)"
                    )

                resultados.append(
                    {
                        "carpeta": grupo,
                        "archivo": foto.name,
                        "predicciones": predicciones,
                    }
                )
            except Exception as exc:
                print(f"\nERROR en {foto.name}: {exc}")
                resultados.append(
                    {"carpeta": grupo, "archivo": foto.name, "error": str(exc)}
                )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for grupo in ("conocidos", "desconocidos"):
        grupo_resultados = [r for r in resultados if r["carpeta"] == grupo]
        exitosos = [r for r in grupo_resultados if "predicciones" in r]
        print(f"{grupo}: {len(exitosos)}/{len(grupo_resultados)} procesadas")

    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")
    print("NOTA: Top 1 es la clase mas probable, NO confirma que este registrada.")
    print("La CNN aun NO tiene umbral validado para rechazar desconocidos.")


if __name__ == "__main__":
    ejecutar()
