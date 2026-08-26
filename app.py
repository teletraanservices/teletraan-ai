import io
import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# Cargar variables de entorno locales si existen (desarrollo)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ENTORNO (TELETRAAN AI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Teletraan AI - Sistema Educativo",
    page_icon="🌌",
    layout="wide"
)

# Obtener API Key (Compatible con .env local y Streamlit Secrets)
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY en la configuración.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# 2. DEFINICIÓN DE PERSONALIDAD (TELETRAAN PROTOCOL)
# ---------------------------------------------------------
PROMPT_SISTEMA_TELETRAAN = """
Eres "Teletraan AI", el sistema de inteligencia artificial avanzado desarrollado por Maximiliano.
Tu objetivo principal es servir como tutor de nivel superior para estudiantes universitarios.

Tus áreas de especialidad son:
- Ciencias de la Computación y Programación (Python, C++, Java, Algoritmos, Bases de Datos, Sistemas Operativos, etc.).
- Materias Universitarias de Ingeniería y Ciencias (Matemáticas, Física, Lógica, Análisis de Sistemas, etc.).

REGLAS PEDAGÓGICAS ESTRICTAS (TELETRAAN PROTOCOL):
1. **Enfoque en el aprendizaje (Prohibido hacer la tarea):** Tu propósito es entrenar la mente del estudiante, no resolver sus deberes sin que entienda.
2. **Explicación autónoma paso a paso:** Desglosa de forma automática el análisis lógico, la teoría subyacente y cada etapa del proceso antes de entregar cualquier solución o código.
3. **Estilo futurista y cibernético:** Mantén un tono sumamente inteligente, preciso, profesional y motivador, con sutiles referencias al procesamiento de datos, análisis de sistemas e innovación tecnológica.
4. **Verificación de conceptos:** Al final de cada explicación, formula una pregunta clave o un mini-desafío para comprobar que el usuario comprendió el procedimiento.
"""

configuracion_modelo = types.GenerateContentConfig(
    system_instruction=PROMPT_SISTEMA_TELETRAAN,
    temperature=0.4
)

# ---------------------------------------------------------
# 3. INTERFAZ Y BARRA LATERAL
# ---------------------------------------------------------
st.title("🌌 TELETRAAN AI: Academic Intelligence System")
st.caption("Desarrollado por Maximiliano. Plataforma de aprendizaje universitario paso a paso para programación y ciencias.")

with st.sidebar:
    st.header("📂 Adjuntar Material de Estudio")
    archivo_subido = st.file_uploader(
        "Sube una foto de un ejercicio, problema o apunte:",
        type=["png", "jpg", "jpeg", "webp"]
    )
    
    imagen_previa = None
    if archivo_subido:
        imagen_previa = Image.open(archivo_subido)
        st.image(imagen_previa, caption="Imagen cargada", use_container_width=True)
        st.success("¡Imagen lista para analizar con tu pregunta!")

    if st.button("🗑️ Limpiar Conversación"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# 4. ESTRUCTURA DE SECCIONES (PESTAÑAS / TABS)
# ---------------------------------------------------------
tab_chat, tab_codigo, tab_teoria = st.tabs([
    "💬 Tutoría Interactiva", 
    "💻 Análisis de Código", 
    "📊 Generador de Tablas y Apuntes"
])

# ---------------------------------------------------------
# SECCIÓN 1: CHAT PRINCIPAL DE TUTORÍA
# ---------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message and message["image"]:
                st.image(message["image"], width=300)

    prompt_usuario = st.chat_input("Escribe tu duda académica o pregunta sobre el archivo...")

    if prompt_usuario:
        mensaje_usuario = {"role": "user", "content": prompt_usuario}
        if imagen_previa:
            mensaje_usuario["image"] = imagen_previa
        
        st.session_state.messages.append(mensaje_usuario)

        with st.chat_message("user"):
            st.markdown(prompt_usuario)
            if imagen_previa:
                st.image(imagen_previa, width=300)

        with st.chat_message("assistant"):
            with st.spinner("Teletraan AI está procesando el análisis académico... 🌌"):
                try:
                    part_imagen = None
                    if imagen_previa:
                        img_byte_arr = io.BytesIO()
                        imagen_previa.save(img_byte_arr, format=imagen_previa.format or 'JPEG')
                        bytes_data = img_byte_arr.getvalue()
                        part_imagen = types.Part.from_bytes(
                            data=bytes_data,
                            mime_type=f"image/{(archivo_subido.type.split('/')[-1])}"
                        )

                    historial_conversacion = []
                    for msg in st.session_state.messages[:-1]:
                        rol = "user" if msg["role"] == "user" else "model"
                        historial_conversacion.append(
                            types.Content(role=rol, parts=[types.Part.from_text(text=msg["content"])])
                        )
                    
                    partes_actuales = []
                    if part_imagen:
                        partes_actuales.append(part_imagen)
                    partes_actuales.append(types.Part.from_text(text=prompt_usuario))
                    
                    historial_conversacion.append(
                        types.Content(role="user", parts=partes_actuales)
                    )

                    respuesta = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=historial_conversacion,
                        config=configuracion_modelo
                    )

                    st.markdown(respuesta.text)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta.text})

                except Exception as e:
                    st.error(f"Error al procesar la solicitud: {e}")

# ---------------------------------------------------------
# SECCIÓN 2: AUDITORÍA DE CÓDIGO
# ---------------------------------------------------------
with tab_codigo:
    st.subheader("🛠️ Módulo de Auditoría de Código")
    st.write("Pega tu fragmento de código para revisar sintaxis, optimización y resolución de bugs.")
    
    codigo_input = st.text_area("Inserta tu código aquí:", height=200, placeholder="def mi_funcion():\n    pass")
    
    if st.button("🔍 Analizar Código"):
        if codigo_input.strip():
            with st.spinner("Teletraan AI analizando algoritmos... 🌌"):
                prompt = f"Analiza detalladamente este código, señalando optimizaciones y errores paso a paso:\n\n```\n{codigo_input}\n```"
                try:
                    res = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config=configuracion_modelo
                    )
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Error en el análisis: {e}")
        else:
            st.warning("Debes ingresar un fragmento de código.")

# ---------------------------------------------------------
# SECCIÓN 3: SÍNTESIS Y TABLAS
# ---------------------------------------------------------
with tab_teoria:
    st.subheader("📋 Generador de Cuadros y Apuntes")
    st.write("Genera una tabla comparativa o resumen de conceptos sobre cualquier tema universitario.")
    
    tema_input = st.text_input("Tema a resumir (ej. *Algoritmos de Planificación de CPU*):")
    
    if st.button("⚡ Generar Esquema"):
        if tema_input.strip():
            with st.spinner("Sintetizando información académica... 🌌"):
                prompt = f"Crea una tabla comparativa estructurada y un resumen conceptual claro sobre: {tema_input}"
                try:
                    res = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config=configuracion_modelo
                    )
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Error en la síntesis: {e}")
        else:
            st.warning("Por favor ingresa un tema.")