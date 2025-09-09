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
import os
import aiohttp
import json
import io
import tempfile
from pydub import AudioSegment

# Giả định bạn có file scoring.py
try:
    from utils.scoring import score_transcription
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

# !!! QUAN TRỌNG: SỬA LỖI Ở ĐÂY !!!
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
#  HELPER FUNCTION
# ------------------
async def process_audio_file(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    waveform, sample_rate = torchaudio.load(tmp_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    return waveform.to(device), 16000, tmp_path

# ------------------------------------
#  API ENDPOINTS
# ------------------------------------
@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    try:
        waveform, sample_rate = await process_audio_file(file)
        
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"🎤 Transcription for practice: '{transcription}'")
        
        result = score_transcription(transcription, target)
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
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"transcription": transcription.strip()}
    except Exception as e:
        print(f"🔥 Error in /transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(chat_message: ChatMessage):
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
                response.raise_for_status() # This will raise an error for 4xx/5xx responses
                result = await response.json()
        
        if result and result.get('candidates'):
            generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
            if generated_text:
                return {"response": generated_text.strip()}
        
        raise HTTPException(status_code=500, detail="Gemini API returned an invalid response format.")
    except aiohttp.ClientResponseError as e:
        # Bắt lỗi cụ thể từ API call
        print(f"🔥 Gemini API Error: Status {e.status}, Message: {e.message}")
        raise HTTPException(status_code=502, detail=f"Failed to communicate with the AI service. Reason: {e.message}")
    except Exception as e:
        print(f"🔥 Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tongue-twisters")
async def get_tongue_twisters():
    twisters = [
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    ]
    return {"tongue_twisters": twisters}

@app.get("/topics")
async def get_topics():
    # This endpoint now calls the generator function directly.
    return await generate_topics()

@app.get("/generate-topics")
async def generate_topics():
    try:
        payload = { 
            "contents": [{"parts": [{"text": "Generate 5 English pronunciation topics for learners, from easy to hard. For each topic, create a title and 5-10 example sentences. The response must be a valid JSON array of objects, where each object has 'id', 'title', and 'sentences' keys."}]}], 
            "generationConfig": {"responseMimeType": "application/json"} 
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API_URL, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    print(f"🔥 Gemini API Error for Topics: Status {response.status}, Body: {error_body}")
                    raise HTTPException(status_code=502, detail=f"AI service failed to generate topics. Status: {response.status}")
                result = await response.json()

        if result and result.get('candidates'):
            generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
            if generated_text:
                return {"topics": json.loads(generated_text)}
        
        raise HTTPException(status_code=500, detail="Gemini API returned an invalid format for topics.")
    except Exception as e:
        print(f"🔥 Error in /generate-topics endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------
#  RUN THE APP
# ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

