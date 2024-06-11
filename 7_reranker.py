# 重排序
from FlagEmbedding import LayerWiseFlagLLMReranker
from tqdm import tqdm
import json
import numpy as np


# 初始化 reranker，指定设备
reranker = LayerWiseFlagLLMReranker("model/bge-reranker-v2-minicpm-layerwise", use_fp16=True)
with open(
    "datasets/AQA-test-public/pid_to_title_abs_update_filter.json",
    "r",
    encoding="utf-8",
) as file:
    papers = json.load(file)
with open("datasets/test_data_process_by_llama3.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

with open("datasets/llama3_train_bge_top100_test.txt", "r", encoding="utf-8") as file:
    data = file.readlines()

pids_list = [line.strip().split(",") for line in data]


count = len(questions)

for i in tqdm(range(0, count), desc="Process line", total=count):
    q = questions[i]["question"] + questions[i]["explain_by_llama3"]
    queries = [
        [
            q,
            f"paper title: {papers[pid]['title']}\n paper abstract:\n {papers[pid]['abstract']}",
        ]
        for pid in pids_list[i]
    ]

    score = reranker.compute_score(queries, cutoff_layers=[28])

    score = np.array(score)
    idxs = np.argsort(score)[::-1][:20]

    search_result = []
    for j in idxs:
        search_result.append(pids_list[i][j])

    with open("datasets/res_llama3_mix_model_55_LayerWise_reranker_test.txt", "a") as file:
        file.write(",".join(search_result) + "\n")
