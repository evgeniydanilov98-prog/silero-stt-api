from fastapi import FastAPI, File, UploadFile, HTTPException
import whisper
import tempfile
import os

# Инициализация FastAPI приложения
app = FastAPI(title="Whisper STT API")

# Глобальная переменная для модели (изначально пустая)
model = None

# Функция для загрузки модели Whisper
def load_model():
    global model
    print("Загрузка модели Whisper... Это займет несколько секунд.")
    # 'tiny' - самая быстрая и легкая модель. Есть 'base', 'small', 'medium', 'large'
    model = whisper.load_model("tiny")
    print("Модель Whisper успешно загружена!")
    return model

@app.get("/", tags=["Health Check"])
async def health_check():
    """Простой эндпоинт для проверки, что сервис работает."""
    return {"status": "ok", "message": "Whisper STT API is running"}

@app.post("/stt", tags=["Speech-to-Text"])
async def speech_to_text(file: UploadFile = File(...)):
    """
    Принимает аудиофайл, возвращает транскрибированный текст.
    Поддерживаемые форматы: wav, mp3, ogg, flac и другие, которые понимает ffmpeg.
    """
    # Проверяем, загружена ли модель. Если нет - загружаем.
    global model
    if model is None:
        print("Модель еще не загружена, загружаю сейчас...")
        load_model()

    if not file:
        raise HTTPException(status_code=400, detail="Файл не был предоставлен.")

    # Создаем временный файл для сохранения аудио
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_audio:
        try:
            audio_content = await file.read()
            tmp_audio.write(audio_content)
            tmp_audio_path = tmp_audio.name

            # Отправляем файл на транскрибацию в Whisper
            result = model.transcribe(tmp_audio_path, language="ru")
            
            return {"text": result["text"]}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при обработке аудио: {str(e)}")
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
