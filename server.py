# ------------------
#  IMPORTS
# ------------------
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
from pydub import AudioSegment # <-- IMPORT PYDUB
import io # <-- IMPORT IO ĐỂ XỬ LÝ DỮ LIỆU TRONG BỘ NHỚ

# Giả định bạn có file scoring.py
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

# Tải model từ Hugging Face Hub là cách làm ổn định nhất
# Việc tải từ Google Drive URL trực tiếp trong code là không khả thi
try:
    print("⬇️  Loading ASR model from Hugging Face Hub...")
    asr_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", task="transcribe", language="en")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    asr_model.to(device)
    asr_model.eval()
    print("✅ ASR model loaded successfully.")
except Exception as e:
    print(f"❌ Critical error loading ASR model: {e}")
    asr_model = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ")

if GEMINI_API_KEY == "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ":
    print("⚠️ WARNING: Using a hardcoded placeholder Gemini API Key.")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"



# ------------------
# ------------------
#  PYDANTIC MODELS
# ------------------
class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []
    system_prompt_override: Optional[str] = None

# (Các model khác cho analyze-pronunciation có thể thêm vào đây nếu cần)

# ------------------
#  HELPER FUNCTION (Sử dụng Pydub, không cần FFmpeg)
# ------------------
async def process_audio_file(file: UploadFile):
    """
    Reads an uploaded audio file, converts it to a standard WAV format in memory using pydub,
    and then loads it with torchaudio.
    """
    try:
        print(f"📄 Received file: {file.filename} ({file.content_type})")
        # Đọc nội dung file vào bộ nhớ
        file_content = await file.read()
        
        # Tạo một đối tượng file-like trong bộ nhớ
        audio_stream = io.BytesIO(file_content)

        # Sử dụng pydub để đọc audio từ stream, bất kể định dạng gốc
        print("🔄 Converting audio using pydub...")
        audio_segment = AudioSegment.from_file(audio_stream)
        
        # Chuẩn hóa audio: 1 kênh (mono), sample rate 16kHz
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_channels(1)
        
        print("✅ Audio standardized to 16kHz mono WAV format.")

        # Xuất audio đã chuẩn hóa ra một stream WAV trong bộ nhớ
        wav_stream = io.BytesIO()
        audio_segment.export(wav_stream, format="wav")
        wav_stream.seek(0) # Rất quan trọng: Đưa con trỏ về đầu stream

        # Tải waveform từ stream WAV bằng torchaudio
        waveform, sample_rate = torchaudio.load(wav_stream)
        
        print(f"🎧 Loaded audio with torchaudio. Sample rate: {sample_rate}, Shape: {waveform.shape}")
        
        # Không cần file tạm, không cần xóa
        return waveform.to(device), sample_rate

    except Exception as e:
        print(f"❌ Error processing audio file: {e}")
        # Ném ra lỗi để endpoint có thể bắt và xử lý
        raise ValueError(f"Could not process audio file '{file.filename}'. Reason: {e}")


# ------------------------------------
#  API ENDPOINTS
# ------------------------------------

@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    
    try:
        # Hàm mới không trả về tmp_path nữa
        waveform, sample_rate = await process_audio_file(file)
        
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


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    try:
        waveform, sample_rate = await process_audio_file(file)
        
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

# Các endpoint khác giữ nguyên vì chúng không xử lý audio
@app.post("/chat")
async def chat(chat_message: ChatMessage):
    # ... code giữ nguyên
    try:
        prompt_parts = []
        for msg in chat_message.history:
            if msg.get('user_message') and msg.get('chatbot_response'):
                prompt_parts.append({"role": "user", "parts": [{"text": msg['user_message']}]})
                prompt_parts.append({"role": "model", "parts": [{"text": msg['chatbot_response']}]})
        prompt_parts.append({"role": "user", "parts": [{"text": chat_message.message}]})
        system_instruction = chat_message.system_prompt_override or "Your name is Lilly. You are a friendly English tutor..."
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
    # ... code giữ nguyên
    twisters = ["She sells seashells...", "Peter Piper..."]
    return {"tongue_twisters": twisters}

@app.get("/topics")
async def get_topics():
    # ... code giữ nguyên
    return await generate_topics()

@app.get("/generate-topics")
async def generate_topics():
    # ... code giữ nguyên
    try:
        payload = { "contents": [{"parts": [{"text": "Generate 5 English pronunciation topics..."}]}], "generationConfig": {"responseMimeType": "application/json"} }
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

# ------------------
#  RUN THE APP
# ------------------
if __name__ == "__main__":
    import uvicorn
    # Port được lấy từ biến môi trường, phù hợp cho Railway
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
