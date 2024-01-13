import os
import json

def load_gsm8k_data(data_path, split):
    data = {"question":[], "answer":[]}
    with open(data_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = json.loads(line)
            data["question"].append(line["question"])
            data["answer"].append(line["answer"].replace("####", "FINAL ANSWER:"))
    return data

