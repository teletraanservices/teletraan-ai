import os

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Teletraan AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Error del sistema: GEMINI_API_KEY no está configurada en las variables de entorno.",
        )

    genai.configure(api_key=api_key)

    # Intenta el modelo configurado y después los fallbacks en orden.
    preferred_model = os.getenv("GEMINI_MODEL", "").strip()
    candidate_models = [
        model_name
        for model_name in [preferred_model, "gemini-1.5-flash", "gemini-1.5-pro"]
        if model_name
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(request.message)
            return {"reply": response.text, "model_used": model_name}
        except Exception as error:
            last_error = f"Falló modelo [{model_name}]: {error}"

    raise HTTPException(
        status_code=500,
        detail=(
            "Error al conectar con Gemini tras intentar varios modelos. "
            f"Detalle: {last_error}"
        ),
    )


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(ROOT_DIR, "PUBLIC")

if os.path.exists(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


@app.get("/")
async def serve_frontend():
    index_file = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"error": "No se encontró index.html"}
