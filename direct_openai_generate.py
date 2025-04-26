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
from tqdm import tqdm
import jsonlines
import json
import asyncio
from glob import glob
import random


DEBUG = False
save_dir = "./data"

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

def scan_cache(data_dir):
    print(f"Scanning cache directory: {data_dir}")
    cache_files = glob(os.path.join(data_dir, "*.wav"))
    cache_files = set([os.path.basename(f) for f in cache_files])
    print(f"Found {len(cache_files)} cache files.")
    return cache_files


def format_prompt(prompt: str, voice: str) -> str:
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    return tokenizer.decode(all_input_ids[0])

async def generate_audio_stream(prompt: str, voice: str, file_path: str) -> None:
    """Async version of audio generation with non-blocking operations"""
    start_time = time.time()
    total_frames = 0
    total_bytes = 0
    chunk_count = 0
    all_audio_data = bytearray() if DEBUG else None
    
    try:
        formatted_prompt = format_prompt(prompt, voice)
        response = await client.completions.create(
            model="canopylabs/orpheus-3b-0.1-ft",
            prompt=formatted_prompt,
            stream=True,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop=["<|endoftext|>"],
            timeout=600,
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
        
        complete_string = ""
        async for token in token_generator():
            complete_string += token

        return complete_string, file_path
        # with wave.open(file_path, 'wb') as wf:
            # wf.setnchannels(1)
            # wf.setsampwidth(2)
            # wf.setframerate(SAMPLE_RATE)

            # async for audio_chunk in tokens_decoder(token_generator()):
                # if audio_chunk:
                    # chunk_count += 1
                    # total_bytes += len(audio_chunk)
                    # frame_count = len(audio_chunk) // 2  # 16-bit samples
                    # total_frames += frame_count
                    
                    # if DEBUG:
                        # all_audio_data.extend(audio_chunk)
                    
                    # wf.writeframes(audio_chunk)

        # logger.info(f"Processed {chunk_count} chunks ({total_bytes} bytes)")
        # duration = total_frames / SAMPLE_RATE
        # generation_time = time.time() - start_time
        # logger.info(f"Real-time factor: {duration/generation_time:.2f}x")
    except Exception as e:
        logger.error(f"Error in async generation: {str(e)}")
        raise

def generate_requests(data_path: str, save_dir: str, start_index: int = 0, cache_files: set = None, end_index: int = None):
    user_voice = ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]
    select_voice = random.choice(user_voice)
    with jsonlines.open(data_path, 'r') as reader:
        for idx, da in enumerate(reader):
            if end_index is not None and idx >= end_index:
                break
            if idx < start_index:
                continue
            messages = da['messages']
            id = da["id"]
            for round, message in enumerate(messages, 1):
                content = message["content"]
                role = message["role"]
                voice = select_voice if role == "user" else "tara"
                for cont in content:
                    if cont["type"] == "text":
                        save_path = os.path.join(save_dir, f"data_idx_{idx}_order_{round}_{role}.wav")
                        # Don't yield if the file already exists and is in the cache
                        if cache_files is not None and os.path.basename(save_path) in cache_files:
                            continue
                        yield cont["text"], save_path, voice


async def main(args):
    data_path = args.data_path
    save_dir = args.save_dir
    start_index = args.start_index
    end_index = args.end_index
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    cache_files = scan_cache(save_dir)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def _process(text, save_path, voice):
        async with semaphore:
            return await generate_audio_stream(text, voice, save_path)

    tasks = []
    for (text, save_path, voice) in generate_requests(data_path, save_dir, start_index, cache_files, end_index):
        tasks.append(asyncio.create_task(_process(text, save_path, voice)))
    
    pbar = tqdm(total=len(tasks), desc="Processing requests")
    results = {}
    for task in asyncio.as_completed(tasks):
        result = await task
        if result is not None:
            results[result[1]] = result[0]
        pbar.update(1)
    pbar.close()

    with open(os.path.join(save_dir, f"responses_{start_index}_{end_index}.json"), 'w') as f:
        json.dump(results, f)
    

    total_duration = time.time() - start_time
    print(f"\nAll requests completed in {total_duration:.2f} seconds")
    print(f"Average time per request: {total_duration/len(tasks):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate audio files from text using OpenAI API.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Directory to save the generated audio files.")
    parser.add_argument("--concurrency", type=int, default=48, help="Number of concurrent requests.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing.")
    parser.add_argument("--end_index", type=int, default=None, help="End index for processing.")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))

