
# HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling
<p align="center">
  If you find this project useful, please give us a star 🌟.
</p>
<p align="center">
  <img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab" alt="Twitter">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/pytorch-2.5.0-blue" alt="PyTorch">
  <img src="https://img.shields.io/badge/lightning-2.2.1-purple?logo=lightning&logoColor=white" alt="PyTorch Lightning">
    <a href="https://arxiv.org/abs/2507.18897">
    <img src="https://img.shields.io/badge/arXiv-2507.18897-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv:2507.18897"></a>
    <a href="https://github.com/opendilab/HH-Codec/"><img src="https://img.shields.io/github/stars/opendilab/HH-Codec?style=social" alt="GitHub Repo stars"></a>
</p>

## 📚 Algorithm Overview
<p align="center">
  🎉 Discrete Neural Codec With 24 Tokens Per Second (24KHZ) for Spoken Language Modeling!
</p>
<p align="center">
  <img src="https://github.com/opendilab/HH-Codec/blob/main/main.png" width="50%">
</p>

Different color lines indicate the data flow used in inference and only for training. During inference, the audio is processed through the encoder and VQ1 to generate discrete quantization, which is then refined by the MLP. The decoder and fine-tuned BigVGAN subsequently reconstruct the Mel-spectrogram and audio.

## 📚 Experimental Results
<p align="center">
  <img src="https://github.com/opendilab/HH-Codec/blob/main/exp.png" width="50%">
</p>

$N_q$ denotes the number of quantizers. The origin human voice's UTMOS of three dataset (LibriTTS test-other / LibriTTS test-clean / Seed-TTS-eval) is $3.48$ / $4.05$ / $3.57$.}
## ⚙️ Installation
To install HHCodec, follow these steps:
```python
conda create -n hhcodec python=3.10 # it must >3.10 because use bigvgan
conda activate hhcodec
git clone https://github.com/opendilab/HH-Codec.git
cd HH-Codec 
pip install -e .

# Install Dependencies for UTMOS Evaluation
pip install fairseq

# If you encounter conflicts, try:
pip install pip==24.0
```
## 🚀 Train

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

## 🧩 How to use HH-codec 
You can simply use the training set from step 1, the configuration from step 2, and the training script from step 3 to reproduce the results of the model described in the paper with a single run. Since we are still refining the algorithm, an updated set of optimal model weights will be released after the final version of the paper is accepted by the journal.
```python
wav, sr = torchaudio.load(audio_path).to(device))
wav = convert_audio(wav, sr, 24000, 1).unsqueeze(0).unsqueeze(0)  
# Generating discrete codecs
_, _, _, _, quant, _, index = model.encode(audio)
# Get quant from index only
quant = model.quantize.indices_to_codes(index)
# Reconstruct audio
reconstructed_mel, reconstructed_audios = model.decode(quant)
```

## 🌏 Citation
```latex
@article{xue2025hh,
  title={HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling},
  author={Xue, Rongkun and Niu, Yazhe and Hu, Shuai and Yin, Zixin and Yao, Yongqiang and Yang, Jing},
  journal={arXiv preprint arXiv:2507.18897},
  year={2025}
}
```


## 💓 Acknowledgement
This project has been developed partially based on the following pioneering works on GitHub repositories.
We express our profound gratitude for these foundational resources:
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)
- [vocos](https://github.com/gemelo-ai/vocos)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
- [SimVQ](https://github.com/youngsheen/SimVQ)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [moshi](https://github.com/kyutai-labs/moshi)


## 🏷️ License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

