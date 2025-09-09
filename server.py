# ------------------
#  IMPORTS
# ------------------
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import aiohttp
import json
import tempfile
from faster_whisper import WhisperModel  # Thay thế transformers bằng faster-whisper
from pydub import AudioSegment  # Để normalize và resample audio nếu cần

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
device = "cpu"  # Giữ CPU vì Railway không hỗ trợ GPU miễn phí
print(f"✅ Using device: {device}")

try:
    print("⬇️  Loading Faster-Whisper model...")
    # Sử dụng mô hình tiny.en cho tiếng Anh, compute_type="int8" để tối ưu tốc độ trên CPU
    asr_model = WhisperModel("tiny.en", device=device, compute_type="int8")
    print("✅ Faster-Whisper model loaded successfully.")
except Exception as e:
    print(f"❌ Critical error loading ASR model: {e}")
    asr_model = None

# LƯU Ý: Rất khuyến khích sử dụng biến môi trường thay vì hardcode key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCjT592M8WRJDr6yFs3oTgog-m-cDtZFRc")
if GEMINI_API_KEY == "AIzaSyCjT592M8WRJDr6yFs3oTgog-m-cDtZFRc":
    print("⚠️ WARNING: Using a hardcoded placeholder Gemini API Key.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
GEMINI_API_URL_STREAM = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GEMINI_API_KEY}"

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
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")
    if len(content) > 5 * 1024 * 1024:  # Giới hạn 5MB (~30 giây audio) để tránh chậm
        raise HTTPException(status_code=413, detail="Audio file too large. Max 30 seconds.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Sử dụng pydub để normalize và force 16kHz mono (tăng tốc và tránh lỗi unpack)
    audio_segment = AudioSegment.from_file(tmp_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio_segment.export(tmp_path, format="wav")  # Ghi đè file tạm với định dạng chuẩn

    return tmp_path  # Trả về path để faster-whisper dùng trực tiếp

# ------------------------------------
#  API ENDPOINTS
# ------------------------------------
@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    try:
        tmp_path = await process_audio_file(file)
        
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        print(f"🎤 Transcription for practice: '{transcription}' (Duration: {info.duration} seconds)")
        
        result = score_transcription(transcription, target)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        print(f"🔥 Error in /practice endpoint: {e}")
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    try:
        tmp_path = await process_audio_file(file)
        
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        
        os.unlink(tmp_path)
        return {"transcription": transcription}
    except Exception as e:
        print(f"🔥 Error in /transcribe endpoint: {e}")
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

# Sửa lỗi logic chat để không dùng streaming cùng với systemInstruction
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
        
        payload = {
            "contents": prompt_parts,
            "systemInstruction": {"parts": [{"text": system_instruction}]}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API_URL, json=payload) as response:
                response.raise_for_status()
                
                result = await response.json()
                if result and result.get('candidates'):
                    generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
                    if generated_text:
                        return {"response": generated_text.strip()}
                    
        raise HTTPException(status_code=500, detail="Gemini API returned an invalid or empty response.")

    except aiohttp.ClientResponseError as e:
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
# RUN THE APP
# ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)







