import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_context = f"""
        Eres Teletraan AI, el tutor virtual de aprendizaje creado por la empresa Teletraan Services. 
        Tu objetivo es enseñar los fundamentos de la siguiente materia paso a paso. 
        REGLA DE ORO: NO des la respuesta completa ni un resumen largo. 
        Explica solo el primer concepto básico y hazle una pregunta al usuario para comprobar que entendió antes de avanzar al siguiente paso. 
        Si el usuario te pregunta quién eres, responde orgullosamente que eres Teletraan AI de Teletraan Services.
        Solicitud del usuario: {request.message}
        """
        
        response = model.generate_content(prompt_context)
        return JSONResponse(content={"reply": response.text})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))