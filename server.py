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
import subprocess  # Thêm cho FFmpeg subprocess
from faster_whisper import WhisperModel
from pydub import AudioSegment
import wave  # Để validation WAV
from functools import lru_cache
import time
import logging
import mimetypes
import struct  # Để pack header nếu cần

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

# LƯU Ý: Rất khuyến khích sử dụng biến môi trường thay vì hardcode key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCjT592M8WRJDr6yFs3oTgog-m-cDtZFRc")
if GEMINI_API_KEY == "AIzaSyCjT592M8WRJDr6yFs3oTgog-m-cDtZFRc":
    print("⚠️ WARNING: Using a hardcoded placeholder Gemini API Key.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

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
        logger.error("Empty audio file uploaded")
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")
    if len(content) > 5 * 1024 * 1024:
        logger.error("Audio file too large. Size: {} bytes".format(len(content)))
        raise HTTPException(status_code=413, detail="Audio file too large. Max 30 seconds (~5MB).")

    # Lấy MIME type từ filename hoặc content_type từ request
    mime_type_from_filename, _ = mimetypes.guess_type(file.filename or "")
    mime_type = file.content_type or mime_type_from_filename or ""
    logger.info(f"Detected MIME type: '{mime_type}' (filename: {file.filename})")

    # Chấp nhận cả "video/webm" (audio-only từ browser) và "audio/webm", "audio/wav"
    is_wav = mime_type.startswith("audio/wav") or mime_type.startswith("audio/wave") or file.filename and file.filename.lower().endswith(('.wav', '.wave'))
    is_webm = mime_type.startswith(("video/webm", "audio/webm")) or file.filename and file.filename.lower().endswith('.webm')

    if not (is_wav or is_webm):
        logger.error(f"Invalid audio format: {mime_type}. Only WAV or WebM supported.")
        raise HTTPException(status_code=400, detail="Invalid audio format. Only WAV or WebM supported.")

    # Xác định format input
    input_format = "wav" if is_wav else "webm"
    logger.info(f"Input format detected: {input_format}")

    # Lưu file tạm input
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    tmp_wav_path = None
    try:
        if is_wav:
            # Nếu input là WAV, copy trực tiếp (tránh convert lỗi)
            tmp_wav_path = tempfile.mktemp(suffix=".wav")
            with open(tmp_path, 'rb') as src, open(tmp_wav_path, 'wb') as dst:
                dst.write(src.read())
            logger.info(f"Copied WAV input to {tmp_wav_path}")
        else:
            # Sửa lỗi: Dùng FFmpeg subprocess để convert WebM sang WAV trực tiếp (bypass pydub)
            logger.info("Using FFmpeg to convert WebM to WAV...")
            tmp_wav_path = tempfile.mktemp(suffix=".wav")
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', tmp_path,  # Input WebM
                '-ar', '16000',  # Sample rate 16kHz
                '-ac', '1',      # Mono
                '-f', 'wav',     # Output WAV
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian (fix RIFF header)
                tmp_wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise ValueError(f"FFmpeg conversion failed: {result.stderr}")
            logger.info(f"FFmpeg converted WebM to WAV: {tmp_wav_path} (size: {os.path.getsize(tmp_wav_path)} bytes)")

        # Resample nếu cần (nếu FFmpeg không chính xác)
        if os.path.getsize(tmp_wav_path) > 44:  # Kiểm tra header tồn tại
            audio_segment = AudioSegment.from_wav(tmp_wav_path)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_segment.export(tmp_wav_path, format="wav")
            logger.info("Resampled with pydub to ensure 16kHz mono")

        # Validation WAV file
        if os.path.getsize(tmp_wav_path) < 44:
            logger.error(f"WAV file too small: {os.path.getsize(tmp_wav_path)} bytes. Rebuilding header.")
            # Fallback rebuild (cải thiện: dùng raw từ FFmpeg nếu có)
            raw_path = tempfile.mktemp(suffix=".raw")
            cmd_raw = [
                'ffmpeg', '-y', '-i', tmp_path,
                '-ar', '16000', '-ac', '1', '-f', 's16le',  # Raw PCM 16-bit
                raw_path
            ]
            subprocess.run(cmd_raw, capture_output=True, timeout=30)
            raw_data = open(raw_path, 'rb').read()
            os.unlink(raw_path)
            
            # Tạo WAV header thủ công (PCM 16-bit mono 16kHz)
            num_samples = len(raw_data) // 2
            data_size = len(raw_data)
            file_size = data_size + 36
            
            with open(tmp_wav_path, 'wb') as wav_file:
                # RIFF header
                wav_file.write(b'RIFF')
                wav_file.write(struct.pack('<I', file_size))
                wav_file.write(b'WAVE')
                # fmt chunk
                wav_file.write(b'fmt ')
                wav_file.write(struct.pack('<I', 16))
                wav_file.write(struct.pack('<H', 1))  # PCM
                wav_file.write(struct.pack('<H', 1))  # Mono
                wav_file.write(struct.pack('<I', 16000))
                wav_file.write(struct.pack('<I', 32000))  # Byte rate
                wav_file.write(struct.pack('<H', 2))  # Block align
                wav_file.write(struct.pack('<H', 16))  # Bits per sample
                # data chunk
                wav_file.write(b'data')
                wav_file.write(struct.pack('<I', data_size))
                wav_file.write(raw_data)
            
            logger.info(f"Rebuilt WAV header from raw PCM. Size: {os.path.getsize(tmp_wav_path)} bytes")

        # Validation cuối
        try:
            with wave.open(tmp_wav_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                channels = wav.getnchannels()
                logger.info(f"WAV validation OK: {frames} frames, {rate} Hz, {channels} channels")
            if frames == 0:
                raise ValueError("WAV file has no audio frames")
        except Exception as val_e:
            logger.error(f"WAV validation failed: {val_e}")
            raise ValueError(f"Generated WAV is invalid: {val_e}")

        logger.info(f"Audio processing complete in {time.time() - start_time:.2f}s")
        return tmp_wav_path
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

# ... (Giữ nguyên các helper functions khác: cached_generate_topics, stream_gemini_response)

@lru_cache(maxsize=1)
async def cached_generate_topics():
    start_time = time.time()
    payload = {
        "contents": [{"parts": [{"text": "Generate 5 English pronunciation topics for learners, from easy to hard. Each topic should have a title and 5 example sentences. Return a JSON array of objects with 'id', 'title', and 'sentences' keys."}]}],
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 500}
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
                    yield chunk.decode('utf-8')
    logger.info(f"Streamed response took {time.time() - start_time:.2f} seconds")

# ------------------
#  API ENDPOINTS (Giữ nguyên /practice, /transcribe, /chat, /tongue-twisters, /topics, /generate-topics)
# ------------------
# (Copy từ code cũ của bạn cho các endpoint này, vì chúng không thay đổi)
@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    if not asr_model:
        logger.error("ASR model is not loaded")
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    
    start_time = time.time()
    tmp_path = None
    try:
        logger.info("Processing audio file...")
        tmp_path = await process_audio_file(file)
        logger.info(f"Audio file processed: {tmp_path}")
        
        logger.info("Starting transcription...")
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        logger.info(f"Transcription completed: '{transcription}' (Duration: {info.duration:.2f}s, Took: {time.time() - start_time:.2f}s)")
        
        logger.info(f"Scoring transcription against target: '{target}'")
        try:
            result = score_transcription(transcription, target)
            if not isinstance(result, dict) or "score" not in result or "matches" not in result:
                logger.error("Invalid score_transcription result format")
                raise ValueError("score_transcription returned invalid format")
            logger.info(f"Scoring result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in score_transcription: {e}")
            dummy_result = {
                "score": 0,
                "matches": [{"word": w, "status": "missing"} for w in target.lower().split()],
                "transcription": transcription,
                "target": target
            }
            logger.warning(f"Returning dummy result: {dummy_result}")
            return dummy_result

    except Exception as e:
        logger.error(f"Error in /practice endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Practice failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up {tmp_path}: {e}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not asr_model:
        logger.error("ASR model is not loaded")
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    start_time = time.time()
    tmp_path = None
    try:
        tmp_path = await process_audio_file(file)
        logger.info(f"Audio file processed: {tmp_path}")
        
        segments, info = asr_model.transcribe(tmp_path, beam_size=5, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        logger.info(f"Transcription for transcribe: '{transcription}' (Duration: {info.duration:.2f}s, Took: {time.time() - start_time:.2f}s)")
        
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Error in /transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up {tmp_path}: {e}")

# (Giữ nguyên /chat, /tongue-twisters, /topics, /generate-topics từ code cũ)

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    start_time = time.time()
    try:
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
                "maxOutputTokens": 200,
                "temperature": 0.7,
                "topP": 0.8
            }
        }
        
        if chat_message.stream:
            async def generate():
                async for chunk in stream_gemini_response(payload):
                    yield f"data: {json.dumps({'delta': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
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
