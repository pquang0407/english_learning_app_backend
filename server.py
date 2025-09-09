# ------------------
#  IMPORTS
# ------------------
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import os
import aiohttp
import json
import tempfile
from faster_whisper import WhisperModel
from pydub import AudioSegment
from functools import lru_cache
import time
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Giả định bạn có file scoring.py
try:
    from utils.scoring import score_transcription
except ImportError:
    def score_transcription(transcription, target):
        logger.warning("score_transcription not found. Using dummy scoring.")
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
device = "cpu"
logger.info(f"Using device: {device}")

try:
    logger.info("Loading Faster-Whisper model...")
    asr_model = WhisperModel("tiny.en", device=device, compute_type="int8")
    logger.info("Faster-Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Critical error loading ASR model: {e}")
    asr_model = None

# Cập nhật lên gemini-2.5-flash (mới nhất 2025, nhanh hơn với thinking mode)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ")
if GEMINI_API_KEY == "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ":
    logger.warning("Using placeholder Gemini API Key. Set GEMINI_API_KEY in environment.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# ------------------
#  PYDANTIC MODELS
# ------------------
class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []
    system_prompt_override: Optional[str] = None
    stream: bool = False  # Thêm flag cho streaming

# ------------------
#  HELPER FUNCTIONS
# ------------------
async def process_audio_file(file: UploadFile):
    start_time = time.time()
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large. Max 30 seconds.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    audio_segment = AudioSegment.from_file(tmp_path)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio_segment.export(tmp_path, format="wav")
    
    logger.info(f"Audio processing took {time.time() - start_time:.2f} seconds")
    return tmp_path

@lru_cache(maxsize=1)
async def cached_generate_topics():
    start_time = time.time()
    payload = {
        "contents": [{"parts": [{"text": "Generate 5 English pronunciation topics for learners, from easy to hard. Each topic should have a title and 5 example sentences. Return a JSON array of objects with 'id', 'title', and 'sentences' keys."}]}],
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 500}  # Giới hạn output để nhanh hơn
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(GEMINI_API_URL, json=payload, timeout=10) as response:
            if response.status != 200:
                error_body = await response.text()
                logger.error(f"Gemini API Error for Topics: Status {response.status}, Body: {error_body}")
                raise HTTPException(status_code=502, detail=f"AI service failed. Status: {response.status}")
            result = await response.json()

    if result and result.get('candidates'):
        generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
        if generated_text:
            logger.info(f"Topics generation took {time.time() - start_time:.2f} seconds")
            return json.loads(generated_text)
    
    raise HTTPException(status_code=500, detail="Gemini API returned invalid format.")

async def stream_gemini_response(payload: dict) -> AsyncGenerator[str, None]:
    """Stream phản hồi từ Gemini API"""
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(GEMINI_API_URL, json=payload, timeout=10) as response:
            if response.status != 200:
                error_body = await response.text()
                logger.error(f"Gemini API Error: Status {response.status}, Body: {error_body}")
                yield json.dumps({"error": f"AI service failed: {response.status}"})
                return
            async for chunk in response.content.iter_chunked(1024):
                if chunk:
                    yield chunk.decode('utf-8')  # Stream từng chunk
    logger.info(f"Streamed response took {time.time() - start_time:.2f} seconds")

# ------------------
#  API ENDPOINTS
# ------------------
@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    start_time = time.time()
    try:
        tmp_path = await process_audio_file(file)
        
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        logger.info(f"Transcription for practice: '{transcription}' (Duration: {info.duration:.2f}s, Took: {time.time() - start_time:.2f}s)")
        
        result = score_transcription(transcription, target)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        logger.error(f"Error in /practice endpoint: {e}")
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not asr_model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    start_time = time.time()
    try:
        tmp_path = await process_audio_file(file)
        
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        logger.info(f"Transcription for transcribe: '{transcription}' (Duration: {info.duration:.2f}s, Took: {time.time() - start_time:.2f}s)")
        
        os.unlink(tmp_path)
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Error in /transcribe endpoint: {e}")
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    start_time = time.time()
    try:
        # Tối ưu: Giới hạn history để giảm payload (chỉ 3 tin nhắn gần nhất)
        history = chat_message.history[-3:] if chat_message.history else []
        prompt_parts = []
        for msg in history:
            if msg.get('user_message') and msg.get('chatbot_response'):
                prompt_parts.append({"role": "user", "parts": [{"text": msg['user_message']}]})
                prompt_parts.append({"role": "model", "parts": [{"text": msg['chatbot_response']}]})
        prompt_parts.append({"role": "user", "parts": [{"text": chat_message.message}]})
        system_instruction = chat_message.system_prompt_override or "Your name is Lilly. You are a friendly English tutor."
        
        payload = {
            "contents": prompt_parts,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": 200,  # Giới hạn output để nhanh hơn
                "temperature": 0.7,  # Giảm randomness để xử lý nhanh
                "topP": 0.8
            }
        }
        
        if chat_message.stream:
            # Streaming mode
            async def generate():
                async for chunk in stream_gemini_response(payload):
                    yield f"data: {json.dumps({'delta': chunk})}\n\n"  # SSE format cho stream
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Non-streaming (giữ nguyên cho compatibility)
            async with aiohttp.ClientSession() as session:
                async with session.post(GEMINI_API_URL, json=payload, timeout=10) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            if result and result.get('candidates'):
                generated_text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
                if generated_text:
                    logger.info(f"Chat response took {time.time() - start_time:.2f} seconds")
                    return {"response": generated_text.strip()}
            
            raise HTTPException(status_code=500, detail="Gemini API returned invalid response format.")
    except aiohttp.ClientResponseError as e:
        logger.error(f"Gemini API Error: Status {e.status}, Message: {e.message}")
        raise HTTPException(status_code=502, detail=f"Failed to communicate with AI service: {e.message}")
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tongue-twisters")
async def get_tongue_twisters():
    return {
        "tongue_twisters": [
            "She sells seashells by the seashore.",
            "Peter Piper picked a peck of pickled peppers.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        ]
    }

@app.get("/topics")
async def get_topics():
    return {"topics": await cached_generate_topics()}

@app.get("/generate-topics")
async def generate_topics():
    return {"topics": await cached_generate_topics()}

# ------------------
#  RUN THE APP
# ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
