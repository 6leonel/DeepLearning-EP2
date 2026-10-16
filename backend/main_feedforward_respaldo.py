
from pathlib import Path
import json

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras

# ============================================
# 1. CARGAR MODELO Y ARCHIVOS
# ============================================

CARPETA_MODELO = Path(__file__).resolve().parent / "modelo"

modelo = keras.models.load_model(
    CARPETA_MODELO / "modelo_rostros_ep2.keras",
    compile=False
)

with open(
    CARPETA_MODELO / "clases_ep2.json",
    encoding="utf-8"
) as archivo:
    clases = json.load(archivo)

with open(
    CARPETA_MODELO / "preprocesamiento_ep2.json",
    encoding="utf-8"
) as archivo:
    configuracion = json.load(archivo)

# Ordenar las clases por su indice original
clases = sorted(
    clases,
    key=lambda clase: clase["indice"]
)

if len(clases) != modelo.output_shape[-1]:
    raise RuntimeError(
        "Las clases no coinciden con el modelo."
    )

# Umbral provisional definido en Colab
umbral = configuracion["umbral_confianza_provisional"]

# ============================================
# 2. CONFIGURAR FASTAPI
# ============================================

app = FastAPI(
    title="DeepLearning EP2 - Reconocimiento de rostros",
    description="API para identificar personas usando Deep Learning",
    version="1.0.0"
)


# ============================================
# 3. FORMATO DE IMAGEN DESDE ANDROID
# ============================================

class SolicitudImagen(BaseModel):
    imagen: list[list[list[int]]]


# ============================================
# 4. ENDPOINT DE ESTADO
# ============================================

@app.get("/estado")
def verificar_estado():
    return {
        "estado": "funcionando",
        "modelo": "Feedforward",
        "clases": len(clases),
        "entrada_android": [128, 128, 3]
    }


# ============================================
# 5. ENDPOINT DE RECONOCIMIENTO
# ============================================

@app.post("/predecir")
def predecir_persona(solicitud: SolicitudImagen):

    # Convertir la imagen recibida a NumPy
    try:
        matriz = np.asarray(
            solicitud.imagen,
            dtype=np.float32
        )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Matriz de imagen invalida"
        )

    # Validar dimensiones recibidas
    if matriz.shape != (128, 128, 3):
        raise HTTPException(
            status_code=422,
            detail="La imagen debe ser 128x128x3"
        )

    # Validar valores de los pixeles
    if (
        not np.all(np.isfinite(matriz))
        or np.any(matriz < 0)
        or np.any(matriz > 255)
    ):
        raise HTTPException(
            status_code=422,
            detail="Los pixeles deben estar entre 0 y 255"
        )

    # ========================================
    # 6. PREPROCESAR LA IMAGEN
    # ========================================

    # Crear imagen RGB
    imagen_rgb = Image.fromarray(
        matriz.astype(np.uint8)
    )

    # Convertir de RGB a escala de grises
    imagen_gris = imagen_rgb.convert("L")

    # Redimensionar al formato del modelo
    imagen_gris = imagen_gris.resize(
        (64, 64),
        resample=Image.Resampling.BILINEAR
    )

    # Normalizar los pixeles entre 0 y 1
    entrada = np.asarray(
        imagen_gris,
        dtype=np.float32
    ) / 255.0

    # Agregar dimensiones necesarias
    entrada = entrada.reshape(
        1, 64, 64, 1
    )

    # ========================================
    # 7. REALIZAR PREDICCION
    # ========================================

    probabilidades = modelo.predict(
        entrada,
        verbose=0
    )[0]

    indice = int(np.argmax(probabilidades))

    confianza = float(
        probabilidades[indice]
    )

    # ========================================
    # 8. DIAGNOSTICO DE PREDICCIONES
    # ========================================

    print("\n================================")
    print("TOP 5 PREDICCIONES DEL MODELO")
    print("================================")

    mejores = np.argsort(
        probabilidades
    )[-5:][::-1]

    for i in mejores:

        indice_clase = int(i)

        nombre = clases[indice_clase][
            "nombre_visible"
        ]

        porcentaje = float(
            probabilidades[indice_clase]
        ) * 100

        print(
            f"Indice {indice_clase} - "
            f"{nombre}: {porcentaje:.2f}%"
        )

    # Diagnosticar especificamente la persona
    # 27 del grupo 8, correspondiente al indice 200

    indice_esperado = 200

    confianza_esperada = float(
        probabilidades[indice_esperado]
    ) * 100

    print("\nPERSONA ESPERADA:")
    print("Grupo 08 - persona_27")
    print("Indice esperado:", indice_esperado)

    print(
        f"Confianza de la persona esperada: "
        f"{confianza_esperada:.2f}%"
    )

    print("\nRESULTADO PRINCIPAL:")
    print(
        "Persona predicha:",
        clases[indice]["nombre_visible"]
    )

    print(
        f"Confianza maxima: {confianza * 100:.2f}%"
    )

    print(
        f"Umbral configurado: {umbral * 100:.2f}%"
    )

    print("================================\n")

    # ========================================
    # 9. VERIFICAR SI ESTA EN LA BASE
    # ========================================

    if confianza < umbral:

        return {
            "en_base": False,
            "nombre": "No esta en la base de datos",
            "confianza": confianza
        }

    # ========================================
    # 10. DEVOLVER PERSONA IDENTIFICADA
    # ========================================

    return {
        "en_base": True,
        "nombre": clases[indice]["nombre_visible"],
        "confianza": confianza
    }
