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
import asyncio

from orpheus_tts.decoder import tokens_decoder


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
            model="canopylabs/orpheus-tts-0.1-finetune-prod",
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

        with wave.open(file_path, 'wb') as wf:
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

        # logger.info(f"Processed {chunk_count} chunks ({total_bytes} bytes)")
        # duration = total_frames / SAMPLE_RATE
        # generation_time = time.time() - start_time
        # logger.info(f"Real-time factor: {duration/generation_time:.2f}x")
    except Exception as e:
        logger.error(f"Error in async generation: {str(e)}")
        raise

def generate_requests(data_path: str, save_dir: str, start_index: int = 0, end_index: int = None):
    with jsonlines.open(data_path, 'r') as reader:
        for idx, da in enumerate(reader):
            if end_index is not None and idx >= end_index:
                break
            if idx < start_index:
                continue
            messages = da['messages']
            id = da["id"]
            for message in messages:
                content = message["content"]
                for cont in content:
                    if cont["type"] == "text":
                        save_path = os.path.join(save_dir, f"data_idx_{idx}_{id}_assistant.wav")
                        yield cont["text"], save_path


async def main(args):
    data_path = args.data_path
    save_dir = args.save_dir
    start_index = args.start_index
    end_index = args.end_index
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()

    semaphore = asyncio.Semaphore(args.concurrency)

    async def _process(text, save_path):
        async with semaphore:
            return await generate_audio_stream(text, "tara", save_path)

    tasks = []
    for (text, save_path) in generate_requests(data_path, save_dir, start_index, end_index):
        tasks.append(asyncio.create_task(_process(text, save_path)))
    
    pbar = tqdm(total=len(tasks), desc="Processing requests")
    for task in asyncio.as_completed(tasks):
        await task
        pbar.update(1)
    pbar.close()
    

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

