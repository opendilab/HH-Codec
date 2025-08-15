# Learned from: https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/scripts/hubert_rep_extract.py
import argparse
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from hhcodec.util import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    rep_dir = os.environ.get("REP_PATH")
    if rep_dir is None:
        raise ValueError("Environment variable REP_PATH is not set")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )

    parser.add_argument(
        "--exts",
        type=str,
        default="wav,mp3,flac",
        help="Comma-separated list of audio file extensions.",
    )

    args = parser.parse_args()
    exts = args.exts.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 16000
    segment_size = 65536

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval().to(device)

    target_layer = "avg"
    dataset_path = Path(args.dataset_path)

    file_list = [
        str(file)
        for ext in exts
        for path in [dataset_path]
        for file in path.glob(f"**/*.{ext}")
    ]

    seed_everything(args.seed)
    random.shuffle(file_list)

    print(f"A total of {len(file_list)} samples will be processed.")
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list), total=len(file_list)):
            wav_24k, sample_rate = librosa.load(audio_file, sr=24000, mono=1 == 1)
            wav_24k = torch.as_tensor(wav_24k)
            if wav_24k.size(-1) < segment_size:
                wav_24k = torch.nn.functional.pad(
                    wav_24k, (0, segment_size - wav_24k.size(-1)), "constant"
                )
                wav_16k = torchaudio.functional.resample(wav_24k, 24000, 16000)
            else:
                wav_16k = torchaudio.functional.resample(wav_24k, 24000, 16000)
            input_values = feature_extractor(
                wav_16k.squeeze(0), sampling_rate=16000, return_tensors="pt"
            ).input_values
            output = model(input_values.to(device), output_hidden_states=True)
            rep = torch.mean(torch.stack(output.hidden_states), axis=0)

            if str(dataset_path) in audio_file:
                rep_file = (
                    audio_file.replace(
                        str(dataset_path), f"{rep_dir}/{args.dataset_name}"
                    ).split(".")[0]
                    + ".hubert.npy"
                )
            rep_sub_dir = "/".join(rep_file.split("/")[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())

            train_list = os.path.join(rep_dir, f"{args.dataset_name}.txt")
            with open(train_list, "a+", encoding="utf-8") as f:
                f.write(f"{audio_file}\t{rep_file}\n")
