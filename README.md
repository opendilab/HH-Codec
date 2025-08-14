
# HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling
🎉 Discrete Neural Codec With 24 Tokens Per Second (24KHZ) for Spoken Language Modeling!

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.0-blue)
![wandb](https://img.shields.io/badge/wandb-0.16.6-orange?logo=wandb&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/lightning-2.2.1-purple?logo=lightning&logoColor=white)


## Installation
To install HHCodec, follow these steps:
```python
conda create -n hhcodec python=3.10 # it must >3.10 because use bigvgan
conda activate hhcodec
git clone https://github.com/opendilab/HH-Codec.git
cd HH-Codec 
pip install -e .

#if you want to eval by UTMOS
pip install pip==24.0
pip install fairseq
```
## Train

### Step 1: Prepare the Training Dataset
Ensure your dataset is preprocessed by following the instructions in [`dataset`](dataset)

### Step 2: Modify Configuration Files
Before starting training, update the configuration settings
```python
# Open and modify the following file "configs/train.yaml"
# Adjust parameters such as:
# - log settings
# - train_path
# - save_dir
# - device (e.g., CPU/GPU)
```

### Step 3: Start Training
Once the dataset is prepared and the configuration is set, launch the training process:
```python
cd HH-Codec
python train.py fit --config configs/train.yaml
```

## How to use HH-codec 
You can simply use the training set from part 1, the configuration from part 2, and the training script from part 3 to reproduce the results of the model described in the paper with a single run. Since we are still refining the algorithm, an updated set of optimal model weights will be released after the final version of the paper is accepted by the journal.
```python
wav, sr = torchaudio.load(audio_path).to(device))
wav = convert_audio(wav, sr, 24000, 1).unsqueeze(0).unsqueeze(0) #eg. [B,1,24000]
# Generating discrete codecs
_, _, _, _,quant,_,index = model.encode(audio)
# Get quant from index only
quant = model.quantize.indices_to_codes(index)
# Reconstruct audio from raw wav
reconstructed_mel, reconstructed_audios = model.decode(quant)
```

## 
## Acknowledgement
The HHCodec codebase is adapted from the following repositories:
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)
- [vocos](https://github.com/gemelo-ai/vocos)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
- [SimVQ](https://github.com/youngsheen/SimVQ)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [moshi](https://github.com/kyutai-labs/moshi)

A huge thanks to the authors of these projects for their outstanding contributions! 🎉

