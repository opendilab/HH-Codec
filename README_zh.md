# HH-Codec: 用于语音语言大模型的高压缩高保真离散编解码器（Tokenizer/Codec）

<p align="center">
  中文 | <a href="README.md"> English</a>
</p>

<p align="center">
  如果您觉得这个项目有用，请给我们一个 GitHub 星标 🌟。
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

## 📚 算法概述
<p align="center">
  🎉 用于语音语言大模型的离散编解码器，24KHz 采样率下每秒只需 24个 token！
</p>
<p align="center">
  <img src="https://github.com/opendilab/HH-Codec/blob/main/main.png" width="50%">
</p>

不同颜色的线分别表示推理中使用的数据流和仅用于训练的数据流。在推理过程中，输入音频通过编码器和 VQ1 处理生成离散量化结果，然后由 MLP 进行细化。再通过解码器和特殊微调后的 BigVGAN 重建梅尔频谱图和音频。

## 📚 实验结果
<p align="center">
  <img src="https://github.com/opendilab/HH-Codec/blob/main/exp.png" width="50%">
</p>

其中 $N_q$ 表示量化器的数量。三个测试数据集（LibriTTS test-other / LibriTTS test-clean / Seed-TTS-eval）中原始人声 UTMOS 指标分别为 $3.48$ / $4.05$ / $3.57$。

## ⚙️ 安装
要安装 HHCodec，请按照以下步骤操作：
```python
conda create -n hhcodec python=3.10 # 必须大于3.10，因为使用了bigvgan
conda activate hhcodec
git clone https://github.com/opendilab/HH-Codec.git
cd HH-Codec 
pip install -e .

# 安装UTMOS评估的依赖
pip install fairseq

# 如果遇到冲突，请尝试：
pip install pip==24.0
```

## 🚀 训练

### 步骤1：准备训练数据集
确保您的数据集已按照 [`dataset`](dataset) 中的说明进行预处理。
运行脚本后，将在 REP_PATH 生成文件：
```
dataset/Hubert/libritts_train_clean_100.txt
```
每行将原始音频文件路径映射到其对应的 HuBERT 嵌入位置。

以下数据集需要按此方式处理才能达到论文中声明的指标：
- [LibriSpeech](http://www.openslr.org/12)  
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2651)  
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)  
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)

有关更详细的数据集准备说明，请参阅 [dataset/README](dataset/README_zh.md)。

### 步骤2：修改配置文件
在开始训练之前，更新配置设置
```python
# 打开并修改以下文件 "config/train_with_8gpu.yaml"
# 调整参数，例如：
# - 日志设置
# - 训练路径 "dataset/Hubert/libritts_train_clean_100.txt"
# - 保存目录
# - 设备（例如，CPU/GPU）
```

### 步骤3：开始训练
一旦数据集准备就绪且配置设置完成，启动训练过程的命令如下：
```python
cd HH-Codec
python train.py fit --config configs/train.yaml
```

## 🧩 如何使用HH-codec
您可以简单地使用步骤1中的训练集、步骤2中的配置和步骤3中的训练脚本，来运行复现论文中描述的模型结果。由于我们仍在完善算法，在论文最终版本被期刊接受后，将开源一系列最新的最优模型权重，希望我们设计的语音 tokenizer 能够助力更多的衍生工作。
```python
wav, sr = torchaudio.load(audio_path).to(device))
wav = convert_audio(wav, sr, 24000, 1).unsqueeze(0).unsqueeze(0)  
# 生成离散编码结果
_, _, _, _, quant, _, index = model.encode(audio)
# 从索引获取对应量化后的值
quant = model.quantize.indices_to_codes(index)
# 从量化后的结果重建音频
reconstructed_mel, reconstructed_audios = model.decode(quant)
```

## 🌏 引用
```latex
@article{xue2025hh,
  title={HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling},
  author={Xue, Rongkun and Niu, Yazhe and Hu, Shuai and Yin, Zixin and Yao, Yongqiang and Yang, Jing},
  journal={arXiv preprint arXiv:2507.18897},
  year={2025}
}
```

## 💓 致谢
本项目部分基于 GitHub 上的以下开源工作扩展开发。
我们对这些基础资源表示深切感谢：
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)
- [vocos](https://github.com/gemelo-ai/vocos)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
- [SimVQ](https://github.com/youngsheen/SimVQ)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [moshi](https://github.com/kyutai-labs/moshi)

## 🏷️ 许可证
本仓库中的所有代码均采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可证。 