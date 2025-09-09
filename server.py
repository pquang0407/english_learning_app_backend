from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
import aiohttp
import json
import re
import random

# Giả định bạn có file scoring.py tại app/backend/utils/scoring.py
try:
    from app.backend.utils.scoring import score_transcription
except ImportError:
    def score_transcription(transcription, target):
        print("Warning: 'score_transcription' not found. Using dummy scoring.")
        from difflib import SequenceMatcher
        score = SequenceMatcher(None, transcription.lower(), target.lower()).ratio() * 100
        matches = [{"word": w, "status": "correct" if w in transcription.lower().split() else "missing"} for w in target.lower().split()]
        return {"score": int(score), "matches": matches, "transcription": transcription, "target": target}

# ------------------
#  APP INITIALIZATION
# ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
#  MODEL LOADING & CONFIG
# ------------------
device = torch.device("cpu")
print(f"✅ Using device: {device}")

# Tải model từ Hugging Face Hub, cách này đáng tin cậy hơn
try:
    print("⬇️  Loading ASR model from Hugging Face Hub...")
    asr_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", task="transcribe", language="en")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    asr_model.to(device)
    asr_model.eval()
    print("✅ ASR model loaded successfully.")
except Exception as e:
    print(f"❌ Critical error loading ASR model: {e}")
    # Nếu không tải được model, ứng dụng không thể hoạt động
    # Bạn có thể thêm logic để thoát hoặc xử lý lỗi này
    asr_model = None

# Gemini API configuration
# LƯU Ý: Rất khuyến khích sử dụng biến môi trường thay vì hardcode key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ")
if GEMINI_API_KEY == "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ":
    print("⚠️ WARNING: Using a hardcoded placeholder Gemini API Key.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# ------------------
#  PYDANTIC MODELS
# ------------------
class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []
    system_prompt_override: Optional[str] = None

# ------------------
#  HELPER FUNCTION (ĐÃ SỬA LỖI)
# ------------------
async def process_audio_file(file: UploadFile):
    # Lấy phần mở rộng từ tên tệp gốc (ví dụ: .webm, .wav)
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    
    # Tạo tệp tạm thời với đúng phần mở rộng
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    print(f"📄 Saved uploaded file to temporary path: {tmp_path} ({len(content)} bytes)")

    try:
        # Tải file âm thanh bằng torchaudio
        waveform, sample_rate = torchaudio.load(tmp_path)
        print(f"🎧 Loaded audio. Original sample rate: {sample_rate}, Shape: {waveform.shape}")

        # Chuẩn hóa audio: mono, 16kHz
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            print(f"🔄 Resampled audio to 16000 Hz. New shape: {waveform.shape}")

        return waveform.to(device), 16000, tmp_path
    except Exception as e:
        # Xóa tệp tạm nếu có lỗi xảy ra
        os.remove(tmp_path)
        print(f"❌ Error processing audio file: {e}")
        raise e

# ------------------------------------
#  API ENDPOINTS (ĐÃ CẬP NHẬT)
# ------------------------------------

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    tmp_path = None
    try:
        waveform, sample_rate, tmp_path = await process_audio_file(file)
        
        print("🧠 Performing transcription...")
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"🎤 Transcription result: '{transcription.strip()}'")
        return {"transcription": transcription.strip()}
    except Exception as e:
        print(f"🔥 Error in /transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    tmp_path = None
    try:
        waveform, sample_rate, tmp_path = await process_audio_file(file)
        
        print(f"🧠 Performing practice scoring. Target: '{target}'")
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(f"🎤 Transcription for practice: '{transcription}'")
        result = score_transcription(transcription, target)
        print(f"📊 Scoring result: {result}")
        return result
    except Exception as e:
        print(f"🔥 Error in /practice endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        prompt_parts = []
        for msg in chat_message.history:
            if msg.get('user_message') and msg.get('chatbot_response'):
                prompt_parts.append({"role": "user", "parts": [{"text": msg['user_message']}]})
                prompt_parts.append({"role": "model", "parts": [{"text": msg['chatbot_response']}]})
        
        prompt_parts.append({"role": "user", "parts": [{"text": chat_message.message}]})
        
        system_instruction = chat_message.system_prompt_override or "Your name is Lilly. You are a friendly English tutor. Your responses must be short and in English. Do not use special characters. If the user's sentence has errors, provide a corrected version and a simple explanation. If correct, continue the conversation."
        
        payload = {"contents": prompt_parts, "systemInstruction": {"parts": [{"text": system_instruction}]}}

        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API_URL, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
        
        if result and result.get('candidates'):
            generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
            if generated_text:
                return {"response": generated_text.strip()}
        
        return {"response": "Sorry, I couldn't generate a response."}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}"}

@app.get("/tongue-twisters")
async def get_tongue_twisters():
    twisters = [
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "Red lorry, yellow lorry.", "A proper copper coffee pot.",
        "I scream, you scream, we all scream for ice cream.",
        "Unique New York, you know you need unique New York."
    ]
    return {"tongue_twisters": twisters}

@app.get("/generate-topics")
async def generate_topics():
    try:
        payload = {
            "contents": [{"parts": [{"text": "Generate 5 English pronunciation topics for learners, from easy to hard. For each topic, create a title and 5-10 example sentences. The response must be a valid JSON array of objects, where each object has 'id', 'title', and 'sentences' keys."}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API_URL, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
        if result and result.get('candidates'):
            generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
            if generated_text:
                return {"topics": json.loads(generated_text)}
        return {"error": "Failed to generate topics."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@app.get("/topics")
async def get_topics():
    return await generate_topics()

# ------------------
#  RUN THE APP
# ------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)




