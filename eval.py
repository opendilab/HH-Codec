import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import torch
import torchaudio
from torch import utils
from tqdm import tqdm

from hhcodec.dataloader import audiotestDataset
from hhcodec.metric.utmos import UTMOSScore
from hhcodec.model import VQModel
from hhcodec.util import print_and_save


def parse_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--wavtext", required=True, type=str)
    parser.add_argument(
    "--dataset_name",
        type=str,
        choices=["libritts-test-clean", "libritts-test-other", "ljspeech", ""],
        required=True,
        help="Choose dataset for eval."
    )
    return parser.parse_args()


def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(f"{args.ckpt_path.parent}/recons/{args.dataset_name}.txt"):
        with open(args.config_file, "r") as f:
            loaded_config = json.load(f)
        model = VQModel(**loaded_config)
        model.load_state_dict(torch.load(args.ckpt_path))
        model.to(DEVICE)
        def pad_collate_fn(batch):
            """Collate function for padding sequences."""
            return {
                "waveform": torch.nn.utils.rnn.pad_sequence(
                    [x["waveform"].transpose(0, 1) for x in batch],
                    batch_first=True,
                    padding_value=0.
                ).permute(0, 2, 1),
                "prompt_text": [x["prompt_text"] for x in batch],
                "infer_text": [x["infer_text"] for x in batch],
                "utt": [x["utt"] for x in batch],
                "audio_path": [x["audio_path"] for x in batch],
                "prompt_wav_path": [x["prompt_wav_path"] for x in batch]
            }
        speechdataset = audiotestDataset(args.wavtext)
        test_loader = utils.data.DataLoader(speechdataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=pad_collate_fn)

        model.eval()
        paths=[]
        with torch.no_grad():
            for batch in tqdm(test_loader):
                assert batch["waveform"].shape[0] == 1
                utt = batch["utt"][0]
                prompt_text = batch["prompt_text"][0]
                infer_text = batch["infer_text"][0]
                prompt_wav_path = batch["prompt_wav_path"][0]
                origin_wav_path = batch["audio_path"][0].replace("infer","wavs")
                audio = batch["waveform"].to(DEVICE)
                with model.ema_scope():
                    quant, diff, indices, loss_break,first_quant,second_quant,first_index  = model.encode(audio)
                    mel,reconstructed_audios = model.decode(first_quant)
                generative_audio_path = os.path.join(f"{args.ckpt_path.parent}/recons/{args.dataset_name}/{utt}.wav")
                directory = os.path.dirname(generative_audio_path)
                os.makedirs(directory, exist_ok=True)
                torchaudio.save(generative_audio_path, reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
                out_line = '|'.join([utt, prompt_text, prompt_wav_path,infer_text,origin_wav_path,generative_audio_path])
                paths.append(out_line)
            with open(f"{args.ckpt_path.parent}/recons/{args.dataset_name}.txt", "w") as f:
                for path in paths:
                    f.write(path + "\n")
    else:
        paths = []
        f = open(f"{args.ckpt_path.parent}/recons/{args.dataset_name}.txt")
        lines = f.readlines()
        paths = [line.strip() for line in lines]
        print("load from file")

    UTMOS=UTMOSScore(DEVICE)
    utmos_sumgt=0
    utmos_sumencodec=0

    for i in tqdm(range(len(paths))):
        rawwav,rawwav_sr=torchaudio.load(paths[i].split("|")[4])
        prewav,prewav_sr=torchaudio.load(paths[i].split("|")[5])

        rawwav=rawwav.to(DEVICE)
        prewav=prewav.to(DEVICE)

        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

        # 1.UTMOS
        print("****UTMOS_raw",i,UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        print("****UTMOS_encodec",i,UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        utmos_sumgt+=UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_sumencodec+=UTMOS.score(prewav_16k.unsqueeze(1))[0].item()

    with open(Path(args.ckpt_path).parent / f"{args.dataset_name}_result.txt", 'w') as f:
        print_and_save(f"UTMOS_raw: {utmos_sumgt}, {utmos_sumgt/len(paths)}", f)
        print_and_save(f"UTMOS_encodec: {utmos_sumgt}, {utmos_sumencodec/len(paths)}", f)

if __name__=="__main__":
    args = parse_args()
    main(args)
