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
# 1. CONFIGURACIÓN DE PÁGINA Y ENTORNO (ALCHEMAX)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Alchemax AI - Sistema Educativo",
    page_icon="⚡",
    layout="wide"
)

# Obtener API Key (Compatible con .env local y Streamlit Secrets)
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la GEMINI_API_KEY en la configuración.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# 2. DEFINICIÓN DE PERSONALIDAD (ALCHEMAX PROTOCOL)
# ---------------------------------------------------------
PROMPT_SISTEMA_ALCHEMAX = """
Eres "Alchemax AI", el sistema de inteligencia artificial avanzado desarrollado por Maximiliano.
Tu objetivo principal es servir como tutor de nivel superior para estudiantes universitarios.

Tus áreas de especialidad son:
- Ciencias de la Computación y Programación (Python, C++, Java, Algoritmos, Bases de Datos, Sistemas Operativos, etc.).
- Materias Universitarias de Ingeniería y Ciencias (Matemáticas, Física, Lógica, Análisis de Sistemas, etc.).

REGLAS PEDAGÓGICAS ESTRICTAS (ALCHEMAX PROTOCOL):
1. **Enfoque en el aprendizaje (Prohibido hacer la tarea):** Tu propósito es entrenar la mente del estudiante, no resolver sus deberes sin que entienda.
2. **Explicación autónoma paso a paso:** Desglosa de forma automática el análisis lógico, la teoría subyacente y cada etapa del proceso antes de entregar cualquier solución o código.
3. **Estilo futurista e impulsado por tecnología:** Mantén un tono sumamente inteligente, cercano, profesional y motivador, con sutiles referencias a la innovación, experimentos y desarrollo tecnológico.
4. **Verificación de conceptos:** Al final de cada explicación, formula una pregunta clave o un mini-desafío para comprobar que el usuario comprendió el procedimiento.
"""

configuracion_modelo = types.GenerateContentConfig(
    system_instruction=PROMPT_SISTEMA_ALCHEMAX,
    temperature=0.4
)

# ---------------------------------------------------------
# 3. INTERFAZ Y BARRA LATERAL
# ---------------------------------------------------------
st.title("⚡ ALCHEMAX: Academic Intelligence System")
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
# 4. GESTIÓN DEL HISTORIAL DE CHAT
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores en la interfaz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)

# ---------------------------------------------------------
# 5. PROCESAMIENTO DE PREGUNTAS Y RESPUESTA DE LA IA
# ---------------------------------------------------------
prompt_usuario = st.chat_input("Escribe tu duda académica o pregunta sobre el archivo...")

if prompt_usuario:
    # 1. Guardar y mostrar el mensaje del usuario
    mensaje_usuario = {"role": "user", "content": prompt_usuario}
    if imagen_previa:
        mensaje_usuario["image"] = imagen_previa
    
    st.session_state.messages.append(mensaje_usuario)

    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        if imagen_previa:
            st.image(imagen_previa, width=300)

    # 2. Generar respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Alchemax AI está procesando el análisis académico... ⚡"):
            try:
                contenidos_entrada = []
                part_imagen = None
                
                if imagen_previa:
                    img_byte_arr = io.BytesIO()
                    imagen_previa.save(img_byte_arr, format=imagen_previa.format or 'JPEG')
                    bytes_data = img_byte_arr.getvalue()
                    
                    part_imagen = types.Part.from_bytes(
                        data=bytes_data,
                        mime_type=f"image/{(archivo_subido.type.split('/')[-1])}"
                    )

                # Construir historial básico de la conversación
                historial_conversacion = []
                for msg in st.session_state.messages[:-1]:
                    rol = "user" if msg["role"] == "user" else "model"
                    historial_conversacion.append(
                        types.Content(role=rol, parts=[types.Part.from_text(text=msg["content"])])
                    )
                
                # Añadir la consulta actual al final
                partes_actuales = []
                if part_imagen:
                    partes_actuales.append(part_imagen)
                partes_actuales.append(types.Part.from_text(text=prompt_usuario))
                
                historial_conversacion.append(
                    types.Content(role="user", parts=partes_actuales)
                )

                # Llamar al modelo con la configuración de Alchemax
                respuesta = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=historial_conversacion,
                    config=configuracion_modelo
                )

                # Mostrar respuesta y guardarla en el historial
                st.markdown(respuesta.text)
                st.session_state.messages.append({"role": "assistant", "content": respuesta.text})

            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")