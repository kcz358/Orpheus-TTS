from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import logging
import sys
import time
import os
import wave
import torch
import io
from snac import SNAC
import numpy as np
import torch
import asyncio
import queue

import threading

model_lock = threading.Lock()
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)

def convert_to_audio(multiframe, count):
  frames = []
  if len(multiframe) < 7:
    return
  
  codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

  num_frames = len(multiframe) // 7
  frame = multiframe[:num_frames*7]

  for j in range(num_frames):
    i = 7*j
    if codes_0.shape[0] == 0:
      codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
    else:
      codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

    if codes_1.shape[0] == 0:
      
      codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    else:
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    
    if codes_2.shape[0] == 0:
      codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
    else:
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

  codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
  if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
    return

  with torch.inference_mode():
    audio_hat = model.decode(codes)
  
  audio_slice = audio_hat[:, :, 2048:4096]
  detached_audio = audio_slice.detach().cpu()
  audio_np = detached_audio.numpy()
  audio_int16 = (audio_np * 32767).astype(np.int16)
  audio_bytes = audio_int16.tobytes()
  return audio_bytes

def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    last_token = token_string[last_token_start:]
    
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None
  

async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:           
        token = turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                        buffer_to_proc = buffer[-28:]
                        loop = asyncio.get_event_loop()
                        audio_samples = await loop.run_in_executor(
                            None, 
                            lambda: convert_to_audio(buffer_to_proc, count)
                        )
                        if audio_samples is not None:
                            yield audio_samples


def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None) 

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()

DEBUG = False

TEMPERATURE = 0.9
TOP_P = 0.8
MAX_TOKENS = 15000
REPETITION_PENALTY = 1.0
SAMPLE_RATE = 24000
VLLM_REMOTE = "http://localhost:8000/v1"
VLLM_KEY = "EMPTY"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orpheus TTS API",
    description="API for Orpheus text-to-speech that mimics OpenAI's TTS endpoint",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "Orpheus TTS API",
        "version": "0.1.0",
        "description": "API for Orpheus text-to-speech that mimics OpenAI's TTS endpoint",
        "endpoints": ["/v1/audio/speech"]
    }


tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
'''
client = OpenAI(
    base_url=VLLM_REMOTE,
    api_key=VLLM_KEY  # dummy key, required but not used
)
'''

# Initialize async OpenAI client
client = AsyncOpenAI(
    base_url=VLLM_REMOTE,
    api_key=VLLM_KEY  # dummy key
)


class SpeechRequest(BaseModel):
    model: str = "orpheus-1-ft"  
    input: str
    voice: Literal["leah", "jess", "leo", "dan", "mia", "zac", "zoe", "tara"] = "tara"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = "wav"
    speed: Optional[float] = 1.0

def format_prompt(prompt: str, voice: str) -> str:
  
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    return tokenizer.decode(all_input_ids[0])

async def generate_audio_stream(prompt: str, voice: str):
    """Async version of audio generation with non-blocking operations"""
    start_time = time.time()
    total_frames = 0
    total_bytes = 0
    chunk_count = 0
    all_audio_data = bytearray() if DEBUG else None
    
    try:
        formatted_prompt = format_prompt(prompt, voice)
        response = await client.completions.create(
            model="canopylabs/orpheus-tts-0.1-finetune-prod",
            prompt=formatted_prompt,
            stream=True,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop=["<|endoftext|>"]
        )

        async def token_generator():
            buffer = ""
            async for chunk in response:
                if chunk.choices[0].text is not None:
                    buffer += chunk.choices[0].text
                    while "><custom_token_" in buffer:
                        token_end = buffer.find(">", buffer.find("_token_")) + 1
                        if token_end > 0:
                            complete_token = "<custom" + buffer[buffer.find("_token_"):token_end]
                            if DEBUG:
                                logger.debug(f"Complete token: {complete_token}")
                            yield complete_token
                            buffer = buffer[token_end:]

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)

            async for audio_chunk in tokens_decoder(token_generator()):
                if audio_chunk:
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                    frame_count = len(audio_chunk) // 2  # 16-bit samples
                    total_frames += frame_count
                    
                    if DEBUG:
                        all_audio_data.extend(audio_chunk)
                    
                    wf.writeframes(audio_chunk)

        if DEBUG and all_audio_data:
            with wave.open("debug.wav", 'wb') as debug_file:
                debug_file.setnchannels(1)
                debug_file.setsampwidth(2)
                debug_file.setframerate(SAMPLE_RATE)
                debug_file.writeframes(all_audio_data)

        logger.info(f"Processed {chunk_count} chunks ({total_bytes} bytes)")
        duration = total_frames / SAMPLE_RATE
        generation_time = time.time() - start_time
        logger.info(f"Real-time factor: {duration/generation_time:.2f}x")

        audio_buffer.seek(0)
        return audio_buffer

    except Exception as e:
        logger.error(f"Error in async generation: {str(e)}")
        raise

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Async endpoint handler"""
    try:
        audio_buffer = await generate_audio_stream(request.input, request.voice.lower())
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=orpheus_speech.wav"}
        )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise Exception(status_code=500, detail=str(e))

# Change the uvicorn run command at the bottom to:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30000)