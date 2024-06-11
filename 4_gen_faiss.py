# 生成faiss向量

import json

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = BGEM3FlagModel("model/trianed_bgem3", use_fp16=True)


with open("datasets/AQA-test-public/pid_to_title_abs_update_filter.json", "r", encoding="utf-8") as file:
    papers = json.load(file)

paper_id_list = list(papers.keys())

paper_title_list = [papers[id]["title"] for id in paper_id_list]

paper_abstract_list = [papers[id]["abstract"] for id in paper_id_list]

papers_content = []
for i in range(0, len(paper_title_list)):
    content = f"paper title: {paper_title_list[i]}\n paper abstract: {paper_abstract_list[i]}"
    papers_content.append(content)

paper_embeddings = model.encode(papers_content, batch_size=12, max_length=4096)["dense_vecs"]


dim = paper_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)

index.add(paper_embeddings.astype(np.float32))
faiss.write_index(index, "datasets/faiss/test_paper_content_IndexFlatIP_mixed_model.index")
