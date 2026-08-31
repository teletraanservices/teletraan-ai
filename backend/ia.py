import os
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class ChatRequest(BaseModel):
    message: str
    image_base64: str | None = None

@app.get("/")
def root():
    return {"status": "Teletraan AI Backend Online"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada.")
    
    try:
        sys_instruction = "Eres TELETRAAN AI, el sistema operativo de inteligencia cibernética de Teletraan Services. Responde de forma técnica, precisa y futurista."
        
        contents = [req.message]

        # Procesar imagen si viene adjunta en Base64
        if req.image_base64:
            # Eliminar el encabezado data:image/...;base64, si viene desde el frontend
            if "," in req.image_base64:
                base64_data = req.image_base64.split(",")[1]
            else:
                base64_data = req.image_base64

            image_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(image_bytes))
            contents.append(img)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.7
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))