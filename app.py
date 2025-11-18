from fastapi import FastAPI

# Инициализация FastAPI приложения
app = FastAPI(title="Simple Test API")

@app.get("/")
async def health_check():
    """Простой эндпоинт для проверки, что сервис работает."""
    return {"status": "ok", "message": "API is running"}

@app.post("/stt", tags=["Speech-to-Text"])
async def speech_to_text():
    """
    Временная заглушка. Просто возвращает текст, чтобы проверить связь.
    """
    return {"text": "Это тестовый ответ. Сервис запустился и готов."}
