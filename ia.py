from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. DATOS DE ENTRENAMIENTO (Textos y sus etiquetas)
mensajes_entrenamiento = [
    "Gana dinero rápido y fácil ahora mismo", # Spam (1)
    "Hola, ¿cómo estás? ¿Nos vemos mañana?",  # No spam (0)
    "Premio gigante gratis entra aquí",      # Spam (1)
    "Adjunto el reporte de la reunión de hoy", # No spam (0)
    "Oferta exclusiva compra ya con descuento", # Spam (1)
    "Recuerda comprar la leche al salir"      # No spam (0)
]

# 1 = Spam, 0 = No Spam
etiquetas = [1, 0, 1, 0, 1, 0]

# 2. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# Creamos un 'pipeline' que convierte el texto a números y aplica el algoritmo Naive Bayes
modelo_ia = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenamos a la IA con los datos
modelo_ia.fit(mensajes_entrenamiento, etiquetas)

print("¡IA entrenada con éxito!\n")

# 3. PROBAR LA IA CON MENSAJES NUEVOS (que nunca ha visto)
mensajes_nuevos = [
    "Gana un premio gratis hoy",
    "Hola, ¿a qué hora es la reunión?"
]

predicciones = modelo_ia.predict(mensajes_nuevos)

# 4. MOSTRAR RESULTADOS
for mensaje, prediccion in zip(mensajes_nuevos, predicciones):
    resultado = "SPAM" if prediccion == 1 else "NO SPAM"
    print(f"Mensaje: '{mensaje}' --> Diagnóstico IA: [{resultado}]")