import os
import sys
sys.path.append(os.getcwd())
import glob
from metrics.UTMOS import UTMOSScore
from metrics.ss import SimScore
from metrics.periodicity import calculate_periodicity_metrics
from metrics.wer import WERScore
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from pathlib import Path
from tqdm import tqdm
from taming.data.speech import speechttsTest_en
import importlib
from omegaconf import OmegaConf
import argparse
from torch import utils
metalst="/root/Github/TTS_Tokenizer/data/test_other.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_and_save(message, file):
    print(message)  
    file.write(message + '\n') 

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))
from omegaconf import DictConfig



def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    codebook_size = 8192
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0
    if not os.path.exists(f"{args.ckpt_path.parent}/recons/test_other.txt"):
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
        speechdataset = speechttsTest_en(metalst)
        test_loader = utils.data.DataLoader(speechdataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=pad_collate_fn)
        paths=[]
        with torch.no_grad():
            for batch in tqdm(test_loader):
                assert batch["waveform"].shape[0] == 1
                utt = batch["utt"][0]
                prompt_text = batch["prompt_text"][0]
                infer_text = batch["infer_text"][0]
                prompt_wav_path = batch["prompt_wav_path"][0]
                orgin_wav_path = batch["audio_path"][0].replace("infer","wavs")

                audio = batch["waveform"].to(DEVICE)
                # audio = batch["waveform_16k"].to(DEVICE)


                if model.use_ema:
                    with model.ema_scope():
                        quant, diff, indices, _ = model.encode(audio)

                        reconstructed_audios = model.decode(quant)
                        # mel,reconstructed_audios = model.decode(quant)
                else:
                    quant, diff, indices, _ = model.encode(audio)
                    reconstructed_audios = model.decode(quant)
                    # mel,reconstructed_audios = model.decode(quant)

                for index in indices.flatten():
                    usage[index.item()] += 1
                    
                generative_audio_path = os.path.join(f"{args.ckpt_path.parent}/recons/test_other/{utt}.wav")
                directory = os.path.dirname(generative_audio_path)
                os.makedirs(directory, exist_ok=True)
                torchaudio.save(generative_audio_path, reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
                out_line = '|'.join([utt, prompt_text, prompt_wav_path,infer_text,orgin_wav_path,generative_audio_path])
                paths.append(out_line)
            num_count = sum([1 for key, value in usage.items() if value > 0])
            utilization = num_count / codebook_size
            with open(f"{args.ckpt_path.parent}/recons/test_other.txt", "w") as f:
                for path in paths:
                    f.write(path + "\n")
            with open(Path(args.ckpt_path).parent / "test_other_result.txt", 'w') as f:
                print_and_save(f"utilization: {utilization}", f)


    else:
        paths = []
        f = open(f"{args.ckpt_path.parent}/recons/test_other.txt")
        lines = f.readlines()
        paths = [line.strip() for line in lines]
                        
    
    UTMOS=UTMOSScore(device=DEVICE)
    utmos_sumgt=0
    utmos_sumencodec=0
    pesq_sumpre=0

    for i in tqdm(range(len(paths))):
        rawwav,rawwav_sr=torchaudio.load(paths[i].split("|")[4])
        prewav,prewav_sr=torchaudio.load(paths[i].split("|")[5])
        
        rawwav=rawwav.to(DEVICE)
        prewav=prewav.to(DEVICE)
   
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

        print("****UTMOS_raw",i,UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        print("****UTMOS_encodec",i,UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        utmos_sumgt+=UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_sumencodec+=UTMOS.score(prewav_16k.unsqueeze(1))[0].item()

        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_pesq=rawwav_16k[:,:min_len].squeeze(0)
        prewav_16k_pesq=prewav_16k[:,:min_len].squeeze(0)
        pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        print("****PESQ",i,pesq_score)
        pesq_sumpre+=pesq_score
        
        
    with open(Path(args.ckpt_path).parent / "test_other_result.txt", 'w') as f:
        print_and_save(f"UTMOS_raw: {utmos_sumgt}, {utmos_sumgt/len(paths)}", f)
        print_and_save(f"UTMOS_encodec: {utmos_sumgt}, {utmos_sumencodec/len(paths)}", f)
        print_and_save(f"PESQ: {pesq_sumpre}, {pesq_sumpre/len(paths)}", f)
    
def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--batch_size", default=1, type=int)

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)