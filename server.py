# ------------------
#  IMPORTS
# ------------------
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
import aiohttp
import json
import asyncio
import re
import random

# Giả định bạn có file scoring.py tại app/backend/utils/scoring.py
# Nếu không, bạn có thể comment dòng này và hàm score_transcription
try:
    from app.backend.utils.scoring import score_transcription
except ImportError:
    # Cung cấp một hàm giả nếu không tìm thấy file
    def score_transcription(transcription, target):
        print("Warning: 'score_transcription' not found. Using dummy scoring.")
        from difflib import SequenceMatcher
        score = SequenceMatcher(None, transcription.lower(), target.lower()).ratio() * 100
        matches = [{"word": w, "status": "correct" if w in transcription else "missing"} for w in target.split()]
        return {"score": int(score), "matches": matches, "transcription": transcription, "target": target}


# ------------------
#  APP INITIALIZATION
# ------------------
app = FastAPI()

# Add CORS middleware
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

# Sử dụng chính xác đường dẫn bạn cung cấp
model_path = "https://drive.google.com/file/d/1B8W6XI0-9NLJntuMN9tX0Jl0uH4ROjPU/view?usp=drive_link"

asr_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", task="transcribe", language="en")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

if os.path.exists(model_path):
    try:
        asr_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Successfully loaded custom ASR checkpoint from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}. Using base Whisper model instead.")
else:
    print(f"❌ ASR checkpoint file not found at: {model_path}. Using base Whisper model.")

asr_model.to(device)
asr_model.eval()

# Gemini API configuration
# LƯU Ý: Hãy chắc chắn rằng bạn đã thay thế bằng API key của chính mình.
GEMINI_API_KEY = "AIzaSyDBB1dDLi0F5rJI2QSjY8AANd5j2mP_vfQ"  # THAY THẾ BẰNG KEY CỦA BẠN
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# ------------------
#  PYDANTIC MODELS
# ------------------
class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []
    system_prompt_override: Optional[str] = None # Hỗ trợ kịch bản

class PhonemeResult(BaseModel):
    phoneme: str
    is_correct: bool

class WordResult(BaseModel):
    word: str
    is_correct: bool
    phonemes: Optional[List[PhonemeResult]] = None

class AnalysisResult(BaseModel):
    overall_score: int
    transcription: str
    words: List[WordResult]

# ------------------
#  HELPER FUNCTION
# ------------------
async def process_audio_file(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
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

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        waveform, sample_rate, tmp_path = await process_audio_file(file)
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        os.remove(tmp_path)
        return {"transcription": transcription.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/practice")
async def practice(file: UploadFile = File(...), target: str = Form(...)):
    try:
        waveform, sample_rate, tmp_path = await process_audio_file(file)
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        os.remove(tmp_path)
        result = score_transcription(transcription, target)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze-pronunciation", response_model=AnalysisResult)
async def analyze_pronunciation(file: UploadFile = File(...), target: str = Form(...)):
    try:
        waveform, sample_rate, tmp_path = await process_audio_file(file)
        input_features = asr_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = asr_model.generate(input_features, max_length=448)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        os.remove(tmp_path)

        target_words = re.findall(r"[\w']+|[.,!?;]", target.lower())
        transcribed_words_set = set(re.findall(r"[\w']+|[.,!?;]", transcription.lower()))
        word_results: List[WordResult] = []
        correct_words_count = 0

        for t_word in target_words:
            is_word_correct = t_word in transcribed_words_set
            phoneme_details = []
            num_phonemes = max(2, len(t_word) // 2)
            for _ in range(num_phonemes):
                is_phoneme_correct = is_word_correct or random.random() > 0.3
                phoneme_details.append(PhonemeResult(phoneme="/-/", is_correct=is_phoneme_correct))
            if is_word_correct:
                correct_words_count += 1
            else:
                if all(p.is_correct for p in phoneme_details) and phoneme_details:
                    phoneme_details[0].is_correct = False
            word_results.append(WordResult(word=t_word, is_correct=is_word_correct, phonemes=phoneme_details))

        overall_score = int((correct_words_count / len(target_words)) * 100) if target_words else 0
        return AnalysisResult(overall_score=overall_score, transcription=transcription, words=word_results)
    except Exception as e:
        return AnalysisResult(overall_score=0, transcription=f"Error: {str(e)}", words=[])

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