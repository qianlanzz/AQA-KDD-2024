import json

# 读取原始JSON数据
with open("datasets/train_with_hn.json", "r") as file:
    json_data = json.load(file)

# 准备存储JSONL数据的列表
jsonl_data = []

# 遍历原始JSON数组中的每个元素
for item in json_data:
    # 准备存储正向上下文的列表
    pos_contexts = []
    for ctx in item["positive_ctxs"]:
        # 检查上下文是否为空
        if ctx["title"] is None:
            ctx["title"] = ""
        if ctx["text"] is None:
            ctx["text"] = ""
        # 拼接title和text内容
        pos_contexts.append(ctx["title"] + " " + ctx["text"])

    neg_contexts = []

    for ctx in item["hard_negative_ctxs"]:
        # 检查上下文是否为空
        if ctx["title"] is None:
            ctx["title"] = ""
        if ctx["text"] is None:
            ctx["text"] = ""
        # 拼接title和text内容
        neg_contexts.append(ctx["title"] + " " + ctx["text"])

    for ctx in item["negative_ctxs"]:
        # 检查上下文是否为空
        if ctx["title"] is None:
            ctx["title"] = ""
        if ctx["text"] is None:
            ctx["text"] = ""
        # 拼接title和text内容
        neg_contexts.append(ctx["title"] + " " + ctx["text"])

    # 构建JSONL对象
    jsonl_obj = {"query": item["question"], "pos": pos_contexts, "neg": neg_contexts}

    # 添加JSONL对象到列表中
    jsonl_data.append(jsonl_obj)

# 将JSONL数据写入文件
with open("datasets/train_data_for_bge.jsonl", "w") as file:
    for item in jsonl_data:
        file.write(json.dumps(item) + "\n")
