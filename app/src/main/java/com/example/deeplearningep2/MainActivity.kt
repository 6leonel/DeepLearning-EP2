
package com.example.deeplearningep2

import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

import androidx.core.content.ContextCompat
import com.example.deeplearningep2.ui.theme.DeepLearningEP2Theme

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import org.json.JSONArray
import org.json.JSONObject

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

// ============================================
// 1. CONFIGURACION DE LAS API
// ============================================

// Desde el emulador Android, 100.67.242.82
// permite comunicarse con Windows.

private const val URL_API_ANTIGUA =
    "http://100.67.242.82:8000/predecir"

private const val URL_API_YUNET =
    "http://100.67.242.82:8000/predecir-yunet"

private const val PERMISO_RED_LOCAL =
    "android.permission.ACCESS_LOCAL_NETWORK"

private const val TAMANO_ANTIGUO = 128

private const val MAX_BYTES_FOTOGRAFIA =
    12 * 1024 * 1024

// ============================================
// 2. RESULTADO DEL RECONOCIMIENTO
// ============================================

data class ResultadoReconocimiento(
    val enBase: Boolean,
    val nombre: String,
    val confianza: Double,
    val motivo: String? = null
)

// ============================================
// 3. ACTIVIDAD PRINCIPAL
// ============================================

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContent {
            DeepLearningEP2Theme {
                PantallaReconocimiento()
            }
        }
    }
}

// ============================================
// 4. PERMISO DE RED LOCAL
// ============================================

private fun tienePermisoRedLocal(
    contexto: Context
): Boolean {

    if (Build.VERSION.SDK_INT < 37) {
        return true
    }

    return ContextCompat.checkSelfPermission(
        contexto,
        PERMISO_RED_LOCAL
    ) == PackageManager.PERMISSION_GRANTED
}

// ============================================
// 5. PANTALLA DE RECONOCIMIENTO
// ============================================

@Composable
fun PantallaReconocimiento() {

    val contexto = LocalContext.current

    val scope = rememberCoroutineScope()

    var fotografia by remember {
        mutableStateOf<Bitmap?>(null)
    }

    var uriFotografia by remember {
        mutableStateOf<Uri?>(null)
    }

    var uriCamaraPendiente by remember {
        mutableStateOf<Uri?>(null)
    }

    var usarYuNet by remember {
        mutableStateOf(true)
    }

    var resultado by remember {
        mutableStateOf<ResultadoReconocimiento?>(null)
    }

    var error by remember {
        mutableStateOf<String?>(null)
    }

    var cargando by remember {
        mutableStateOf(false)
    }

    var permisoConcedido by remember {
        mutableStateOf(
            tienePermisoRedLocal(contexto)
        )
    }

    // ========================================
    // SOLICITAR PERMISO
    // ========================================

    val pedirPermiso =
        rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { concedido ->

            permisoConcedido = concedido

            error = if (concedido) {
                null
            } else {
                "Debes permitir el acceso a la red local."
            }
        }

    // ========================================
    // ABRIR CAMARA
    // ========================================

    val abrirCamara =
        rememberLauncherForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { capturaExitosa ->

            val uri = uriCamaraPendiente

            uriCamaraPendiente = null

            if (capturaExitosa && uri != null) {

                try {
                    fotografia = cargarFotografia(
                        contexto,
                        uri
                    )

                    uriFotografia = uri

                    resultado = null
                    error = null

                } catch (e: Exception) {
                    error =
                        "Error al cargar fotografia: ${e.message}"
                }

            } else if (uri != null) {

                runCatching {
                    contexto.contentResolver.delete(
                        uri,
                        null,
                        null
                    )
                }
            }
        }

    // ========================================
    // ABRIR GALERIA
    // ========================================

    val abrirGaleria =
        rememberLauncherForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {

                try {
                    fotografia = cargarFotografia(
                        contexto,
                        uri
                    )

                    uriFotografia = uri

                    resultado = null
                    error = null

                } catch (e: Exception) {
                    error =
                        "Error al abrir imagen: ${e.message}"
                }
            }
        }

    // ========================================
    // INTERFAZ GRAFICA
    // ========================================

    Scaffold(
        modifier = Modifier.fillMaxSize()
    ) { padding ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(
                    rememberScrollState()
                )
                .padding(16.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally,

            verticalArrangement =
                Arrangement.spacedBy(16.dp)
        ) {

            // TITULO

            Text(
                text = "Reconocimiento Facial",
                style =
                    MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "Evaluacion Parcial 2 - Deep Learning"
            )

            Text(
                text = if (usarYuNet) {
                    "CNN MobileNetV2 + YuNet (258 identidades)"
                } else {
                    "CNN anterior (264 clases)"
                },
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "Selecciona una fotografia para identificar a una persona registrada."
            )

            // =================================
            // SELECCION DEL MODELO
            // =================================

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),

                    verticalAlignment =
                        Alignment.CenterVertically,

                    horizontalArrangement =
                        Arrangement.SpaceBetween
                ) {

                    Column(
                        modifier = Modifier.weight(1f)
                    ) {

                        Text(
                            text = "CNN mejorada + YuNet",
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "Desactivar para usar la CNN anterior.",
                            style =
                                MaterialTheme.typography.bodySmall
                        )
                    }

                    Switch(
                        checked = usarYuNet,
                        enabled = !cargando,

                        onCheckedChange = { seleccionado ->

                            usarYuNet = seleccionado

                            resultado = null

                            error = null
                        }
                    )
                }
            }

            // =================================
            // BOTONES CAMARA Y GALERIA
            // =================================

            Row(
                modifier = Modifier.fillMaxWidth(),

                horizontalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Button(
                    onClick = {

                        try {

                            val uri =
                                crearUriCamara(contexto)

                            uriCamaraPendiente = uri

                            abrirCamara.launch(uri)

                        } catch (e: Exception) {

                            error =
                                "No se pudo abrir la camara: ${e.message}"
                        }
                    },

                    enabled = !cargando,

                    modifier = Modifier.weight(1f)
                ) {
                    Text("Camara")
                }

                OutlinedButton(
                    onClick = {
                        abrirGaleria.launch("image/*")
                    },

                    enabled = !cargando,

                    modifier = Modifier.weight(1f)
                ) {
                    Text("Galeria")
                }
            }

            // =================================
            // PERMISO DE RED LOCAL
            // =================================

            if (
                !permisoConcedido &&
                Build.VERSION.SDK_INT >= 37
            ) {

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {

                    Column(
                        modifier = Modifier.padding(16.dp),

                        verticalArrangement =
                            Arrangement.spacedBy(8.dp)
                    ) {

                        Text(
                            text = "Permiso de red local",
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "Android necesita permiso para conectarse con FastAPI."
                        )

                        Button(
                            onClick = {
                                pedirPermiso.launch(
                                    PERMISO_RED_LOCAL
                                )
                            },

                            modifier = Modifier.fillMaxWidth()
                        ) {

                            Text("Permitir red local")
                        }
                    }
                }
            }

            // =================================
            // MOSTRAR FOTOGRAFIA
            // =================================

            fotografia?.let { imagenActual ->

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {

                    Image(
                        bitmap =
                            imagenActual.asImageBitmap(),

                        contentDescription =
                            "Fotografia seleccionada",

                        modifier = Modifier
                            .fillMaxWidth()
                            .height(280.dp),

                        contentScale = ContentScale.Fit
                    )
                }

                // =============================
                // BOTON RECONOCER
                // =============================

                Button(
                    onClick = {

                        permisoConcedido =
                            tienePermisoRedLocal(
                                contexto
                            )

                        if (!permisoConcedido) {

                            pedirPermiso.launch(
                                PERMISO_RED_LOCAL
                            )

                        } else {

                            val uriActual =
                                uriFotografia

                            if (uriActual == null) {

                                error =
                                    "Selecciona una fotografia."

                            } else {

                                cargando = true

                                resultado = null

                                error = null

                                scope.launch {

                                    try {

                                        resultado = if (
                                            usarYuNet
                                        ) {

                                            consultarApiYuNet(
                                                contexto,
                                                uriActual
                                            )

                                        } else {

                                            consultarApiAnterior(
                                                imagenActual
                                            )
                                        }

                                    } catch (e: Exception) {

                                        error =
                                            "Error al consultar FastAPI: ${e.message}"

                                    } finally {

                                        cargando = false
                                    }
                                }
                            }
                        }
                    },

                    enabled = !cargando,

                    modifier = Modifier.fillMaxWidth()
                ) {

                    Text("Reconocer persona")
                }
            }

            // =================================
            // INDICADOR DE CARGA
            // =================================

            if (cargando) {

                CircularProgressIndicator()

                Text(
                    text = "Detectando rostro y consultando la CNN..."
                )
            }

            // =================================
            // RESULTADO DEL RECONOCIMIENTO
            // =================================

            resultado?.let { respuesta ->

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {

                    Column(
                        modifier = Modifier.padding(16.dp),

                        verticalArrangement =
                            Arrangement.spacedBy(8.dp)
                    ) {

                        Text(
                            text = "Resultado del reconocimiento",
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = if (respuesta.enBase) {
                                "Coincidencia propuesta"
                            } else {
                                "Identidad no verificada"
                            },

                            color = if (respuesta.enBase) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            }
                        )

                        Text(
                            text = respuesta.nombre,

                            style =
                                MaterialTheme.typography.titleMedium
                        )

                        Text(
                            text = String.format(
                                Locale.US,
                                "Confianza: %.2f%%",
                                respuesta.confianza * 100.0
                            )
                        )

                        // MOTIVO DE RECHAZO

                        respuesta.motivo?.let { motivo ->

                            when (motivo) {

                                "sin_rostro" -> {
                                    Text(
                                        "YuNet no detecto un rostro."
                                    )
                                }

                                "varios_rostros" -> {
                                    Text(
                                        "Se detectaron varios rostros. Usa una foto individual."
                                    )
                                }
                            }
                        }

                        Text(
                            text =
                                "El umbral del 70% es provisional. " +
                                        "Una confianza alta no garantiza la identidad.",

                            style =
                                MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }

            // =================================
            // MOSTRAR ERRORES
            // =================================

            error?.let { mensaje ->

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {

                    Text(
                        text = mensaje,

                        modifier = Modifier.padding(16.dp),

                        color =
                            MaterialTheme.colorScheme.error
                    )
                }
            }

            // =================================
            // INFORMACION DE LA API
            // =================================

            Text(
                text = if (usarYuNet) {

                    "La fotografia original se envia a FastAPI. " +
                            "YuNet detecta y recorta el rostro."

                } else {

                    "La CNN anterior utiliza una matriz RGB 128x128x3."
                },

                style =
                    MaterialTheme.typography.bodySmall
            )
        }
    }
}

// ============================================
// 6. CREAR ARCHIVO PARA LA CAMARA
// ============================================

private fun crearUriCamara(
    contexto: Context
): Uri {

    val valores = ContentValues().apply {

        put(
            MediaStore.Images.Media.DISPLAY_NAME,
            "ep2_${System.currentTimeMillis()}.jpg"
        )

        put(
            MediaStore.Images.Media.MIME_TYPE,
            "image/jpeg"
        )

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.Q
        ) {

            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                "Pictures/DeepLearningEP2"
            )
        }
    }

    return contexto.contentResolver.insert(
        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
        valores
    ) ?: throw IOException(
        "No se pudo crear el archivo de fotografia."
    )
}

// ============================================
// 7. CARGAR FOTOGRAFIA PARA VISUALIZACION
// ============================================

// Esta funcion solamente reduce la vista previa.
// La imagen original se mantiene para YuNet.

private fun cargarFotografia(
    contexto: Context,
    uri: Uri
): Bitmap {

    if (
        Build.VERSION.SDK_INT >=
        Build.VERSION_CODES.P
    ) {

        val fuente = ImageDecoder.createSource(
            contexto.contentResolver,
            uri
        )

        return ImageDecoder.decodeBitmap(
            fuente
        ) { decoder, info, _ ->

            val ancho = info.size.width

            val alto = info.size.height

            val escala = minOf(
                1.0,
                1024.0 / maxOf(ancho, alto)
            )

            decoder.setTargetSize(
                (ancho * escala).toInt()
                    .coerceAtLeast(1),

                (alto * escala).toInt()
                    .coerceAtLeast(1)
            )

            decoder.allocator =
                ImageDecoder.ALLOCATOR_SOFTWARE
        }
    }

    return contexto.contentResolver
        .openInputStream(uri)?.use { entrada ->

            BitmapFactory.decodeStream(
                entrada
            )

        } ?: throw IOException(
        "No se pudo leer la fotografia."
    )
}

// ============================================
// 8. LEER FOTOGRAFIA ORIGINAL
// ============================================

private fun leerBytesFotografia(
    contexto: Context,
    uri: Uri
): ByteArray {

    val entrada =
        contexto.contentResolver.openInputStream(
            uri
        ) ?: throw IOException(
            "No se pudo abrir la fotografia original."
        )

    return entrada.use { flujo ->

        val salida = ByteArrayOutputStream()

        val buffer = ByteArray(8192)

        var total = 0

        while (true) {

            val leidos = flujo.read(buffer)

            if (leidos == -1) {
                break
            }

            total += leidos

            if (
                total > MAX_BYTES_FOTOGRAFIA
            ) {

                throw IOException(
                    "La fotografia supera los 12 MB."
                )
            }

            salida.write(
                buffer,
                0,
                leidos
            )
        }

        if (total == 0) {

            throw IOException(
                "La fotografia esta vacia."
            )
        }

        salida.toByteArray()
    }
}

// ============================================
// 9. CONSULTAR NUEVA CNN + YUNET
// ============================================

private suspend fun consultarApiYuNet(
    contexto: Context,
    uri: Uri
): ResultadoReconocimiento =
    withContext(Dispatchers.IO) {

        // Leer fotografia sin reducir a 128x128.

        val bytes = leerBytesFotografia(
            contexto,
            uri
        )

        // Convertir imagen a Base64.

        val base64 = Base64.encodeToString(
            bytes,
            Base64.NO_WRAP
        )

        // Construir JSON para FastAPI.

        val cuerpo = JSONObject()
            .put(
                "imagen_base64",
                base64
            )
            .toString()

        // Enviar a la CNN mejorada.

        ejecutarSolicitud(
            URL_API_YUNET,
            cuerpo
        )
    }

// ============================================
// 10. CONSULTAR CNN ANTERIOR
// ============================================

private suspend fun consultarApiAnterior(
    imagen: Bitmap
): ResultadoReconocimiento =
    withContext(Dispatchers.IO) {

        // La CNN antigua necesita 128x128x3.

        val reducida = Bitmap.createScaledBitmap(
            imagen,
            TAMANO_ANTIGUO,
            TAMANO_ANTIGUO,
            true
        )

        val pixeles = IntArray(
            TAMANO_ANTIGUO * TAMANO_ANTIGUO
        )

        reducida.getPixels(
            pixeles,
            0,
            TAMANO_ANTIGUO,
            0,
            0,
            TAMANO_ANTIGUO,
            TAMANO_ANTIGUO
        )

        val matriz = JSONArray()

        for (
        y in 0 until TAMANO_ANTIGUO
        ) {

            val fila = JSONArray()

            for (
            x in 0 until TAMANO_ANTIGUO
            ) {

                val pixel = pixeles[
                    y * TAMANO_ANTIGUO + x
                ]

                val rgb = JSONArray()
                    .put(Color.red(pixel))
                    .put(Color.green(pixel))
                    .put(Color.blue(pixel))

                fila.put(rgb)
            }

            matriz.put(fila)
        }

        val cuerpo = JSONObject()
            .put("imagen", matriz)
            .toString()

        ejecutarSolicitud(
            URL_API_ANTIGUA,
            cuerpo
        )
    }

// ============================================
// 11. CONEXION HTTP CON FASTAPI
// ============================================

private fun ejecutarSolicitud(
    urlApi: String,
    cuerpo: String
): ResultadoReconocimiento {

    val conexion = URL(
        urlApi
    ).openConnection() as HttpURLConnection

    try {

        conexion.requestMethod = "POST"

        conexion.doOutput = true

        conexion.connectTimeout = 15000

        conexion.readTimeout = 120000

        conexion.setRequestProperty(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        // Enviar solicitud.

        conexion.outputStream.use { salida ->

            salida.write(
                cuerpo.toByteArray(
                    Charsets.UTF_8
                )
            )
        }

        // Leer respuesta.

        val codigo = conexion.responseCode

        val texto = if (
            codigo in 200..299
        ) {

            conexion.inputStream
                .bufferedReader()
                .use { it.readText() }

        } else {

            conexion.errorStream
                ?.bufferedReader()
                ?.use { it.readText() }
                ?: "Sin detalles"
        }

        if (
            codigo !in 200..299
        ) {

            val detalle = try {

                JSONObject(texto).optString(
                    "detail",
                    texto
                )

            } catch (_: Exception) {

                texto
            }

            throw IOException(
                "HTTP $codigo: $detalle"
            )
        }

        // Convertir JSON en resultado.

        val json = JSONObject(texto)

        return ResultadoReconocimiento(
            enBase = json.getBoolean(
                "en_base"
            ),

            nombre = json.getString(
                "nombre"
            ),

            confianza = json.getDouble(
                "confianza"
            ),

            motivo = json.optString(
                "motivo",
                ""
            ).ifEmpty {
                null
            }
        )

    } finally {

        conexion.disconnect()
    }
}
