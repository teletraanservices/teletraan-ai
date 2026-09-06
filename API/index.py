import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Teletraan AI Service")

# Habilitar CORS para permitir peticiones desde GitHub Pages o cualquier frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar la API Key de Gemini desde las variables de entorno de Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Estructura de la petición de chat
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no está configurada en el servidor.")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(request.message)
        # Devolvemos 'reply' para que encaje con data.reply de tu script.js
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir los archivos estáticos si se entra directamente a Render
if os.path.exists("PUBLIC"):
    app.mount("/static", StaticFiles(directory="PUBLIC"), name="static")

@app.get("/")
async def serve_frontend():
    if os.path.exists(os.path.join("PUBLIC", "index.html")):
        return FileResponse(os.path.join("PUBLIC", "index.html"))
    return {"message": "Teletraan AI Backend está activo."}