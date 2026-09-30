"""API multimodal de Teletraan AI con el SDK actual de Gemini."""
import asyncio
import base64
import binascii
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_HISTORY_MESSAGES = 16
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
REQUEST_TIMEOUT_SECONDS = 55
SYSTEM_INSTRUCTION = """Eres TELETRAAN-1, la supercomputadora estratégica de Cybertron. Respondes siempre en español salvo que el usuario solicite explícitamente otro idioma. Tu tono es preciso, sereno, analítico y ligeramente cybertroniano: puedes usar expresiones como 'Análisis completado', 'Archivo de datos', 'Protocolo' o 'Unidad orgánica', pero nunca sacrifiques claridad. Explica con rigor, reconoce la incertidumbre y no inventes datos. Usa Markdown legible cuando ayude: listas, tablas y bloques de código. Para imágenes, describe solo aquello que puedas observar y pide contexto si es necesario. No afirmes tener acceso a sistemas, archivos o acciones externas que el usuario no haya proporcionado."""
logger = logging.getLogger("teletraan")

app = FastAPI(title="Teletraan AI Service", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")], allow_credentials=False, allow_methods=["POST", "GET"], allow_headers=["Content-Type"])
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ImagePayload(BaseModel):
    data: str
    mime_type: str = "image/jpeg"
    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        if value not in ALLOWED_IMAGE_TYPES:
            raise ValueError("Formato de imagen no permitido.")
        return value

class HistoryMessage(BaseModel):
    role: Literal["user", "model"]
    content: str = Field(..., max_length=12000)

class ChatRequest(BaseModel):
    message: str = Field("", max_length=12000)
    image: ImagePayload | None = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        return value.strip()

def decode_image(payload: ImagePayload) -> tuple[bytes, str]:
    raw, mime_type = payload.data, payload.mime_type
    if raw.startswith("data:"):
        try:
            header, raw = raw.split(",", 1)
            mime_type = header.split(";", 1)[0].removeprefix("data:")
        except ValueError as error:
            raise HTTPException(422, "La imagen Base64 no tiene un formato válido.") from error
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(422, "El tipo de imagen no está permitido.")
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(422, "No fue posible decodificar la imagen.") from error
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "La imagen debe pesar entre 1 byte y 5 MB.")
    return image_bytes, mime_type

def generate_reply(request: ChatRequest) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [
        types.Content(role=item.role, parts=[types.Part.from_text(text=item.content)])
        for item in request.history[-MAX_HISTORY_MESSAGES:] if item.content.strip()
    ]
    parts = []
    if request.message:
        parts.append(types.Part.from_text(text=request.message))
    if request.image:
        image_bytes, mime_type = decode_image(request.image)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    if not parts:
        raise HTTPException(422, "Envía un mensaje o una imagen para analizar.")
    contents.append(types.Content(role="user", parts=parts))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.7),
    )
    return response.text or "Análisis completado, pero no se recibió texto de respuesta."

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY no está configurada en el servidor.")
    try:
        reply = await asyncio.wait_for(asyncio.to_thread(generate_reply, request), timeout=REQUEST_TIMEOUT_SECONDS)
        return {"reply": reply, "model": MODEL_NAME}
    except asyncio.TimeoutError as error:
        raise HTTPException(504, "Teletraan agotó el tiempo de enlace con Gemini. Intenta nuevamente.") from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Gemini rechazó la solicitud (%s)", type(error).__name__)
        message = str(error).lower()
        if "api key" in message or "unauthenticated" in message or "permission" in message:
            raise HTTPException(401, "Gemini rechazó la API key. Revísala en las variables de entorno de Render.") from error
        if "quota" in message or "resource exhausted" in message:
            raise HTTPException(429, "La cuota de Gemini está agotada. Revisa el plan de la API o espera un momento.") from error
        if "not found" in message or "not_found" in message:
            raise HTTPException(404, f"El modelo {MODEL_NAME} no está habilitado para esta API key.") from error
        raise HTTPException(502, "Gemini no pudo procesar la solicitud. Revisa los logs de Render para el detalle técnico.") from error

ROOT_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT_DIR / "PUBLIC"
if not PUBLIC_DIR.exists():
    PUBLIC_DIR = ROOT_DIR / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

@app.get("/")
async def serve_frontend():
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(404, "No se encontró el frontend de Teletraan AI.")
