from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchaudio
import os
import tempfile

# Инициализация FastAPI приложения
app = FastAPI(title="Silero STT API")

# Глобальные переменные для модели и утилит
model = None
utils = None

# Функция для загрузки модели Silero
def load_model():
    global model, utils
    print("Загрузка модели Silero STT... Это может занять время.")
    # Указываем язык и модель
    language = 'ru'
    model_id = 'v4_ru'
    
    # Загружаем модель и утилиты
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language=language,
                                           device='cpu') # Для Render Free Tier используем CPU
    model.eval() # Переводим модель в режим оценки
    
    # Сохраняем нужные утилиты
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils
    
    print("Модель успешно загружена!")
    return model, (read_batch, split_into_batches, read_audio, prepare_model_input)

# Загружаем модель при старте приложения
model, utils = load_model()

@app.get("/", tags=["Health Check"])
async def health_check():
    """Простой эндпоинт для проверки, что сервис работает."""
    return {"status": "ok", "message": "Silero STT API is running"}

@app.post("/stt", tags=["Speech-to-Text"])
async def speech_to_text(file: UploadFile = File(...)):
    """
    Принимает аудиофайл, возвращает транскрибированный текст.
    Поддерживаемые форматы: wav, mp3, ogg, flac.
    """
    if not file:
        raise HTTPException(status_code=400, detail="Файл не был предоставлен.")

    # Создаем временный файл для сохранения аудио
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        try:
            # Читаем содержимое загруженного файла
            audio_content = await file.read()
            tmp_audio.write(audio_content)
            tmp_audio_path = tmp_audio.name

            # Используем утилиты Silero для чтения и подготовки аудио
            read_audio_func = utils[2]
            prepare_model_input_func = utils[3]
            
            # Читаем аудио и приводим к нужному формату
            wav = read_audio_func(tmp_audio_path)
            input_tensor = prepare_model_input_func([wav])

            # Запускаем распознавание
            output = model(input_tensor)
            
            # Декодируем результат в текст
            # Для простоты используем greedy декодер
            predicted_text = decoder(output[0]).cpu().numpy()[0]
            
            return {"text": str(predicted_text)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при обработке аудио: {str(e)}")
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
