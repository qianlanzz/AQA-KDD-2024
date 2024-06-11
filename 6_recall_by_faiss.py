import json
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm


model = BGEM3FlagModel("model/trianed_bgem3", use_fp16=True)

with open("datasets/AQA-test-public/pid_to_title_abs_update_filter.json", "r", encoding="utf-8") as file:
    papers = json.load(file)

paper_id_list = list(papers.keys())

index = faiss.read_index("datasets/faiss/test_paper_content_IndexFlatIP_mixed_model.index")
res = faiss.StandardGpuResources()
print("Load vector library into gpu.....")
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

with open("datasets/test_data_process_by_llama3.json", "r", encoding="utf-8") as file:
    data = json.load(file)
count = len(data)

for question in tqdm(data, desc="Processing line", total=count):
    content = question["question"] + "\n" + question["explain_by_llama3"]
    embeddings_2 = model.encode(content)["dense_vecs"]
    query = embeddings_2.reshape(1, -1).astype(np.float32)
    k = 100
    D, I = gpu_index.search(query, k)
    search_result = []
    for idx in range(0, k):
        search_result.append(paper_id_list[I[0][idx]])
    with open("datasets/llama3_train_bge_top100_test.txt", "a") as file:
        file.write(",".join(search_result) + "\n")
