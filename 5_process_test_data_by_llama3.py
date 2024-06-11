import transformers
import torch
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result = []

with open("datasets/AQA-test-public/qa_test_wo_ans_new.txt", "r") as file:
    data = file.readlines()
    print(len(file))
    for line in tqdm(file, desc="处理数据"):
        data = json.loads(line)
        question = data["question"]
        body = data["body"]
        result.append({"question": question, "body": body})

with open("datasets/AQA-test-public/qa_test_wo_ans_new.json", "w") as file:
    json.dump(result, file, indent=4)


model_id = "model/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


def process_by_llm(promt, input):
    messages = [
        {
            "role": "system",
            "content": promt,
        },
        {"role": "user", "content": input},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt) :]


with open("datasets/AQA-test-public/qa_test_wo_ans_new.json", "r") as file:
    data = json.load(file)


for q in tqdm(data, desc="处理数据"):
    text = f'Question:\n{q["question"]}\nQuestion details:\n{q["body"]}'
    explain_by_llama3 = process_by_llm(
        "Refine and summarize the given question without leaving out details and proper nouns.Please use a paragraph and don't generate irrelevant content.",
        text,
    )
    explain_by_llama3 = str(explain_by_llama3)
    q["explain_by_llama3"] = explain_by_llama3


with open("datasets/test_data_process_by_llama3.json", "w") as f:
    json.dump(data, f, indent=4)
