
# HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling
🎉 Discrete Neural Codec With 24 Tokens Per Second (24KHZ) for Spoken Language Modeling!

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.0-blue)
![wandb](https://img.shields.io/badge/wandb-0.16.6-orange?logo=wandb&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/lightning-2.2.1-purple?logo=lightning&logoColor=white)

# 🔥 News
We expect to finalize and open-source the training code within two weeks.

## Installation
To install HHCodec, follow these steps:
```python
conda create -n hhcodec python=3.10 # it must >3.10 beacause use bigvgan
conda activate hhcodec
git clone https://github.com/rongkunxue/HH-Codec.git
cd HH-Codec 
pip install -e .

#if you want to eval by UTMOS
pip install pip==24.0
pip install fairseq
```


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

