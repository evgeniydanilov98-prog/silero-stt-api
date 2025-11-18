from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchaudio
import os
import tempfile

# Инициализация FastAPI приложения
app = FastAPI(title="Silero STT API")

# Глобальные переменные для модели и утилит (изначально пустые)
model = None
utils = None

# Функция для загрузки модели Silero (теперь она вызывается отдельно)
def load_model():
    global model, utils
    print("Загрузка модели Silero STT... Это может занять время.")
    language = 'ru'
    model_id = 'v4_ru'
    
    # Загружаем модель и утилиты
    model, decoder, utils_new = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language=language,
                                           device='cpu')
    model.eval()
    
    # Сохраняем нужные утилиты
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils_new
    utils = (read_batch, split_into_batches, read_audio, prepare_model_input)
    
    print("Модель успешно загружена!")
    return model, utils

@app.get("/", tags=["Health Check"])
async def health_check():
    """Простой эндпоинт для проверки, что сервис работает."""
    return {"status": "ok", "message": "Silero STT API is running"}

@app.post("/stt", tags=["Speech-to-Text"])
async def speech_to_text(file: UploadFile = File(...)):
    """
    Принимает аудиофайл, возвращает транскрибированный текст.
    """
    # Проверяем, загружена ли модель. Если нет - загружаем.
    global model
    if model is None:
        print("Модель еще не загружена, загружаю сейчас...")
        load_model()

    if not file:
        raise HTTPException(status_code=400, detail="Файл не был предоставлен.")

    # Создаем временный файл для сохранения аудио
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        try:
            audio_content = await file.read()
            tmp_audio.write(audio_content)
            tmp_audio_path = tmp_audio.name

            # Используем утилиты Silero для чтения и подготовки аудио
            read_audio_func = utils[2]
            prepare_model_input_func = utils[3]
            
            wav = read_audio_func(tmp_audio_path)
            input_tensor = prepare_model_input_func([wav])

            output = model(input_tensor)
            
            # Декодируем результат в текст
            # Для простоты используем greedy декодер
            predicted_text = decoder(output[0]).cpu().numpy()[0]
            
            return {"text": str(predicted_text)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при обработке аудио: {str(e)}")
        finally:
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
