### 数据预处理
要训练 HH-Codec，第一步是下载数据集。我们推荐使用以下训练数据集：

- [LibriSpeech](http://www.openslr.org/12)  
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2651)  
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)  
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)

```shell
cd HH-Codec/dataset
wget https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz
wget https://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz
wget https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz
wget https://openslr.magicdatatech.com/resources/12/test-other.tar.gz
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
wget https://datashare.ed.ac.uk/download/DS_10283_2651.zip
```

您可以直接从原始音频波形中提取语义教师表示到'HH-Codec/dataset'目录。我们提供了一个脚本来演示如何获取HuBERT表示。

```python
cd /root/code/HH-Codec && conda activate codec
export REP_PATH="dataset/Hubert"
python dataset/hubert_mel.py --dataset_name "libritts_train_clean_360" --dataset_path "dataset/LibriSpeech/train-clean-360"
python dataset/hubert_mel.py --dataset_name "libritts_train_clean_100" --dataset_path "dataset/LibriSpeech/train-clean-100"
python dataset/hubert_mel.py --dataset_name "LJspeech" --dataset_path "dataset/LJSpeech-1.1"
python dataset/hubert_mel.py --dataset_name  "" --dataset_path ""
```

运行脚本后，将在以下位置生成文件：
```
dataset/Hubert/libritts_train_clean_100.txt
```
每行将原始音频文件路径映射到其对应的HuBERT嵌入位置。

### 评估数据预处理
```python
cd /root/code/HH-Codec && conda activate codec
python dataset/eval_prepare.py --wavtext "dataset/eval/test_clean.txt" --dataset_path "/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-clean"
python dataset/eval_prepare.py --wavtext "dataset/eval/test_other.txt" --dataset_path "/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-other"
```
