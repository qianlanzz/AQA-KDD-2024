# AQA-KDD-2024


## Prerequisites

* Linux
* Python 3.9
* torch==2.2.1+cu118

## Getting Started


### Installation

Clone this repo.

```shell
git clone https://github.com/qianlanzz/AQA-KDD-2024
```

Please install dependencies by

```shell
pip install -r requirements.txt
```
## datasets
The raw dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1bFM6QM1tv4cz-Vx8VEGp7A?pwd=v2bb) with password v2bb, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/AQA.zip)
The processed data can be downloaded from [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/aqa_train_data_processed.zip).

## model
The bge-m3,bge-reranker-v2-minicpm-layerwise,llama3 models can be obtained from [huggingface](https://huggingface.co/models)

The final fusion model of the bge-m3 and trained models can be found here 
[mixed_model](https://www.alipan.com/s/9fks9UC7hCY)

## Run Code

```shell
#step1 construct training data
python 1_train_data_for_bge.py
#step2 train the bge model
bash 2_finetune_bge.sh
#step3 fusion model
python 3_mix_model.py
#step4 generate the faiss vector for the proceedings
python 4_gen_faiss.py
#step5 clean the test data with llama3
python 5_process_test_data_by_llama3.py
#step6 recall 100 pieces of data with faiss
python 6_recall_by_faiss.py
#step7 use the bge-reranker-v2-minicpm-layerwise model to recall results
python 7_reranker.py
```

## Method Introduction
In this method, bge is used to make embedding vectors for each paper in the proceedings, and the vectors are stored in faiss for vector retrieval. Due to the complexity of the test set and the existence of a large number of web tags, llama3 language model is used to deal with the problem. faiss vector was used to retrieve the most similar 100 paper ids, and then bge-reranker-v2-minicpm-layerwise model was used to reorder the 100 paper ids to get the final result.