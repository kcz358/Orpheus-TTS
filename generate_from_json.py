
import json
import argparse

import torch.distributed as dist
import torch

from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import wave
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import soundfile as sf

SAMPLE_RATE = 24000


def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, world_size


def convert_to_audio(multiframe, count, model, snac_device):
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
    # check that all tokens are between 0 and 4096 otherwise return *
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return

    with torch.inference_mode():
        audio_hat = model.decode(codes)
    return audio_hat

def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        # print("No token found in the string")
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None

def generate_data_by_local_rank(data, local_rank, world_size):
    for idx, (save_path, data_item) in enumerate(data.items()):
        if idx % world_size == local_rank:
            yield save_path, data_item

def write_audio_to_file(audio, save_path):
    sf.write(save_path, audio[0, 0].cpu().numpy(), SAMPLE_RATE)


def split_tokens_to_list(tokens, tokenizer):
    tokens = tokenizer.encode(tokens)
    splited_tokens = []
    count = 0
    for tok in tokens:
        input_id = tokenizer.decode(tok)
        codec = turn_token_into_id(input_id, count)
        if codec is not None:
            splited_tokens.append(codec)
            count += 1
    return splited_tokens


def main(args):
    local_rank, world_size = setup()
    torch.cuda.set_device(local_rank)
    data_path = args.data_path
    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

    snac_device = f"cuda:{local_rank}"
    model = model.to(snac_device)
    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-finetune-prod")


    with open(data_path, 'r') as f:
        data = json.load(f)

    metadata = list(generate_data_by_local_rank(data, local_rank, world_size))
    pbar = tqdm(disable=local_rank != 0, desc="Processing data", total=len(metadata))
    for idx, (save_path, tokens) in enumerate(metadata):
        tokens = split_tokens_to_list(tokens, tokenizer)
        audio_tokens = convert_to_audio(tokens, idx, model, snac_device)
        if audio_tokens is not None:
            write_audio_to_file(audio_tokens, save_path)
        pbar.update(1)
    pbar.close()
    print(f"Rank {local_rank} finished processing data.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio files from text using snac.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated audio files.")
    args = parser.parse_args()
    main(args)
