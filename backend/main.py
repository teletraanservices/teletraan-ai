import os
import time
from google import genai
from google.genai.errors import APIError

# Cargar la clave desde el archivo .env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Inicializar con el modelo solicitado por el SDK
chat = client.chats.create(model="gemini-3.6-flash")

print("--- Chatbot de Inteligencia Artesanal listo (Escribe 'salir' para terminar) ---\n")

while True:
    mensaje_usuario = input("Tú: ")
    
    if mensaje_usuario.lower().strip() in ["salir", "exit", "quit"]:
        print("¡Hasta luego!")
        break

    # Reintento automático en caso de saturación temporal (503)
    exito = False
    intentos = 0
    while not exito and intentos < 3:
        try:
            respuesta = chat.send_message(mensaje_usuario)
            print(f"IA: {respuesta.text}\n")
            exito = True
        except APIError as e:
            if e.code == 503:
                intentos += 1
                print(f"Servidor ocupado. Reintentando ({intentos}/3)...")
                time.sleep(2)
            else:
                print(f"Ocurrió un error: {e}\n")
                break
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}\n")
            break