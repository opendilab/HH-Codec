import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavtext', type=str, help="Output file path", default="dataset/eval/test_clean.txt")
    parser.add_argument('--dataset_path', type=str, help="Dataset path", default="/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-clean")
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3,flac')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    exts = args.exts.split(',')
    file_list = [
        str(file) for ext in exts
        for path in [dataset_path]
        for file in path.glob(f'**/*.{ext}')
    ]

    if not os.path.exists(args.wavtext):
        os.makedirs(os.path.dirname(args.wavtext), exist_ok=True)
    f_w = open(args.wavtext, 'w')
    for i, audio_file in tqdm(enumerate(file_list)):
        file_name = os.path.basename(audio_file)
        utt = os.path.splitext(file_name)[0]
        prompt_text="0"
        prompt_wav="0"
        infer_text=audio_file.replace(".wav", ".normalized.txt")
        with open(infer_text, 'r', encoding='utf-8') as file:
            content = file.read()
        out_line = '|'.join([utt, prompt_text, prompt_wav,content,audio_file])
        f_w.write(out_line + '\n')
    f_w.close()
