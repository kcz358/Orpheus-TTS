import asyncio
import time
import wave
import contextlib
import aiohttp
import jsonlines
from tqdm import tqdm
import os
import argparse

MAX_CONCURRENT = 32

save_dir = "./data"

async def make_tts_request(session, text, index, semaphore):
    url = "http://localhost:30000/v1/audio/speech"
    payload = {
        "input": text,
        "voice": "tara",
        "response_format": "wav"
    }

    start_time = time.time()

    try:
        async with semaphore:  
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    filename = index
                    with open(filename, 'wb') as f:
                        async for chunk in response.content.iter_chunks():
                            if chunk[0]: 
                                f.write(chunk[0])
                    duration = time.time() - start_time
                    print(f"Request {index} completed in {duration:.2f} seconds")
                else:
                    error_text = await response.text()
                    print(f"Request {index} failed with status {response.status}: {error_text}")
    except Exception as e:
        print(f"Error in request {index + 1}: {str(e)}")

def get_wav_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

async def main(args):
    data_path = args.data_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()

    generated_list = []
    with jsonlines.open(data_path, 'r') as reader:
        data = list(reader)
    pbar = tqdm(total=len(data), desc="Processing data")
    for da in data:
        messages = da['messages']
        id = da["id"]
        for message in messages:
            content = message["content"]
            for cont in content:
                if cont["type"] == "text":
                    save_path = os.path.join(save_dir, f"{id}_assistant.wav")
                    generated_list.append((cont["text"], save_path))
        pbar.update(1)
    pbar.close()
    
    assert len(generated_list) == len(data)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [make_tts_request(session, text, i, semaphore) for (text, i) in generated_list]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing requests"):
            await task

    total_duration = time.time() - start_time
    print(f"\nAll requests completed in {total_duration:.2f} seconds")
    print(f"Average time per request: {total_duration/len(generated_list):.2f} seconds")

    # Calculate total audio duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS audio files.")
    parser.add_argument("--data_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Directory to save the generated audio files.")
    args = parser.parse_args()
    asyncio.run(main(args))