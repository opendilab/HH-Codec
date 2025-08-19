### Data Preprocessing
To train the HH-Codec, the first step is to download the dataset. We recommend using the following training datasets:

- [LibriSpeech](http://www.openslr.org/12)  
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2651)  
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)  
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)

```shell
wget https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz
wget https://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz
wget https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz
wget https://openslr.magicdatatech.com/resources/12/test-other.tar.gz
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
wget https://datashare.ed.ac.uk/download/DS_10283_2651.zip
```

You can extract semantic teacher representations directly from raw audio waveforms. We provide a script to demonstrate how to obtain HuBERT representations.

```python
cd /root/code/HH-Codec && conda activate codec
export REP_PATH="/fs-computility/ai-shen/shared/RFTInfra/xrk/dataset/rep"
python dataset/hubert_mel.py --dataset_name "libritts_train_clean_360" --dataset_path ""
python dataset/hubert_mel.py --dataset_name "libritts_train_clean_100" --dataset_path ""
python dataset/hubert_mel.py --dataset_name "LJspeech" --dataset_path "/fs-computility/ai-shen/shared/RFTInfra/xrk/dataset/LJSpeech-1.1"
python dataset/hubert_mel.py --dataset_name 
```

After running the script, a file will be generated at:
```
dataset/LJSpeech.txt
```
Each line maps the original audio file path to its corresponding HuBERT embedding location.



### Eval Data Preprocess
```python
cd /root/code/HH-Codec && conda activate codec
python dataset/eval_prepare.py --wavtext "dataset/eval/test_clean.txt" --dataset_path "/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-clean"
python dataset/eval_prepare.py --wavtext "dataset/eval/test_other.txt" --dataset_path "/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-other"
```